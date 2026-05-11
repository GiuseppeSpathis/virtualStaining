import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
import cv2
from skimage.color import rgb2hed, hed2rgb
from skimage.measure import regionprops

import timm
from timm import layers
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from huggingface_hub import hf_hub_download
from peft import LoraConfig

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from virtual_staining.pixcell_transformer_2d_lora import PixCellTransformer2DModelLoRA
from virtual_staining.resmlp import SimpleMLP
from stardist.models import StarDist2D
from csbdeep.utils import normalize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_images_per_file", type=int, default=300)
    p.add_argument("--input_dir", default=os.path.expanduser("~/giuSpathis/data/he"))
    p.add_argument("--output_base_dir", default=os.path.expanduser("~/giuSpathis/data/pixcellGenIhc"))
    p.add_argument("--key", default="arr_0", help="Key inside the npz")
    p.add_argument("--target", required=True, help="Checkpoint name or path to a custom .pth LoRA file")
    p.add_argument("--flow_target", required=True)
    p.add_argument("--debug_file", type=str, default=None, help="If set, run inference only on this .npz file name")
    p.add_argument("--bgr", action="store_true", help="Interpret input tiles as BGR instead of RGB")
    p.add_argument("--device", default="cuda", help='e.g. "cuda", "cpu"')
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_tokens", type=int, default=16)
    
    # Parametri SDEdit
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=1.2)
    p.add_argument("--flow_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--strength", type=float, default=0.70, help="SDEdit noise strength")
    p.add_argument("--progress_threshold", type=float, default=0.73, help="Soglia di rilascio maschera SDEdit")
    p.add_argument("--hsv_sat_scale", type=float, default=0.4, help="Desaturazione background")
    p.add_argument("--bg_h_preserve", type=float, default=0.2, help="Preservazione background Ematossilina")
    p.add_argument("--stardist_prob_thresh", type=float, default=0.15, help="Soglia StarDist")
    
    # Performance
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--use_tf32", action="store_true")
    p.add_argument("--mmap_npz", action="store_true", default=True)
    p.add_argument("--profile", action="store_true")
    return p.parse_args()


def set_perf_flags(args):
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.backends.cudnn.benchmark = True


def load_uni(device):
    timm_kwargs = {
        "img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24,
        "init_values": 1e-5, "embed_dim": 1536, "mlp_ratio": 2.66667 * 2,
        "num_classes": 0, "no_embed_class": True, "mlp_layer": layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU, "reg_tokens": 8, "dynamic_img_size": True,
    }
    uni_model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    uni_cfg = resolve_data_config(uni_model.pretrained_cfg, model=uni_model)
    uni_model.eval().to(device)
    return uni_model, uni_cfg


def build_transformer_with_lora(device, target: str, num_tokens=16):
    config = {
        "_class_name": "PixCellTransformer2DModel", "_diffusers_version": "0.32.2",
        "_name_or_path": "pixart_1024/transformer", "activation_fn": "gelu-approximate",
        "attention_bias": True, "attention_head_dim": 72, "attention_type": "default",
        "caption_channels": 1536, "caption_num_tokens": num_tokens, "cross_attention_dim": 1152,
        "dropout": 0.0, "in_channels": 16, "interpolation_scale": 2,
        "norm_elementwise_affine": False, "norm_eps": 1e-06, "norm_num_groups": 32,
        "norm_type": "ada_norm_single", "num_attention_heads": 16, "num_embeds_ada_norm": 1000,
        "num_layers": 28, "out_channels": 32, "patch_size": 2, "sample_size": 128,
        "upcast_attention": False, "use_additional_conditions": False,
    }
    transformer = PixCellTransformer2DModelLoRA(**config)
    target_modules = [
        "attn2.add_k_proj", "attn2.add_q_proj", "attn2.add_v_proj",
        "attn2.to_add_out", "attn2.to_k", "attn2.to_out.0",
        "attn2.to_q", "attn2.to_v"
    ]
    transformer.add_adapter(LoraConfig(r=4, lora_alpha=4, init_lora_weights="gaussian", target_modules=target_modules))
    transformer.load_state_dict(torch.load(target, map_location="cpu"), strict=False)
    transformer.eval().to(device)
    return transformer


def build_flow_mlp(device, flow_target: str):
    uni_mlp = SimpleMLP(
        in_channels=1536, time_embed_dim=1024, model_channels=1024,
        bottleneck_channels=1024, out_channels=1536, num_res_blocks=6
    ).to(device)
    uni_mlp.load_state_dict(torch.load(flow_target, map_location="cpu"))
    uni_mlp.eval().to(device)
    return uni_mlp


@torch.no_grad()
def extract_uni_from_batch_tiles_fast(batch_tiles_uint8, uni_model, uni_cfg, device, autocast_dtype, num_tokens=16):
    patch_size = 256 if num_tokens == 16 else 128
    
    x = torch.from_numpy(batch_tiles_uint8).to(device, non_blocking=True).float().div_(255.0)
    x = x.permute(0, 3, 1, 2).contiguous() 

    patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size) 
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, 3, patch_size, patch_size) 

    input_size = uni_cfg["input_size"]  
    patches = F.interpolate(patches, size=input_size[-2:], mode="bicubic", align_corners=False)

    mean = torch.tensor(uni_cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(uni_cfg["std"], device=device).view(1, 3, 1, 1)
    patches = (patches - mean) / std

    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")):
        emb = uni_model(patches) 

    B = batch_tiles_uint8.shape[0]
    return emb.view(B, num_tokens, -1)


@torch.no_grad()
def flow_he_to_ihc(uni_emb_he, uni_mlp, flow_steps, autocast_dtype, num_tokens=16):
    B = uni_emb_he.shape[0]
    x = uni_emb_he.reshape(-1, 1536).float()
    dt = 1.0 / flow_steps
    ts = torch.linspace(1e-3, 1.0 - 1e-3, steps=flow_steps, device=x.device, dtype=torch.float32)

    for t in ts:
        time_tensor = torch.full((num_tokens * B,), t, device=x.device) * 999.0
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(x.is_cuda)):
            dx = uni_mlp(x, time_tensor)
        x = x + dt * dx.float()

    return x.view(B, num_tokens, 1536)


def process_tile_morphology(he_np, stardist_model, hsv_sat_scale, bg_h_preserve, prob_thresh):
    """Esegue StarDist, Pulisce la Maschera, Deconvoluzione Colore e prepara Maschera Latente in un solo passaggio."""
    img_norm = normalize(he_np, 1, 99.8, axis=(0,1,2))
    labels, _ = stardist_model.predict_instances(img_norm, prob_thresh=prob_thresh)
    
    mask_clean = np.zeros_like(labels, dtype=np.float32)
    for prop in regionprops(labels):
        if prop.area < 40 or prop.area > 2000 or prop.eccentricity > 0.85:
            continue
        mask_clean[labels == prop.label] = 1.0
        
    hsv_bg = cv2.cvtColor(he_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_bg[:,:,1] = hsv_bg[:,:,1] * hsv_sat_scale 
    he_neutralized_rgb = cv2.cvtColor(hsv_bg.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    he_float_01 = he_neutralized_rgb.astype(np.float32) / 255.0
    hed = rgb2hed(he_float_01)
    gray = cv2.cvtColor(he_neutralized_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    
    adaptive_retention = np.interp(gray, [0.65, 0.85], [0.40, 0.0])
    final_retention = bg_h_preserve + (1.0 - bg_h_preserve) * adaptive_retention
    
    mask_bg = 1.0 - mask_clean
    hed[:,:,0] = (mask_clean * hed[:,:,0]) + (mask_bg * hed[:,:,0] * final_retention)
    
    he_suppressed_rgb = hed2rgb(hed)
    he_suppressed_rgb = np.clip(he_suppressed_rgb * 255.0, 0, 255).astype(np.uint8)
    
    mask_blur = cv2.GaussianBlur(mask_clean, (5, 5), 0)
    mask_3d = np.expand_dims(mask_blur, axis=-1)
    
    preconditioned_np = (mask_3d * he_np.astype(np.float32)) + ((1.0 - mask_3d) * he_suppressed_rgb.astype(np.float32))
    preconditioned_uint8 = np.clip(preconditioned_np, 0, 255).astype(np.uint8)
    
    return preconditioned_uint8, mask_clean


@torch.no_grad()
def run_hybrid_sdedit_batch(transformer, vae, scheduler, uni_cond, latent_he, mask_latent,
                            seed, total_steps, strength, cfg_scale, autocast_dtype, device, progress_threshold):
    B = uni_cond.shape[0]
    uncond = transformer.caption_projection.uncond_embedding.clone().tile(B, 1, 1).to(device)
    g = torch.Generator(device=device).manual_seed(seed)
    
    scheduler.set_timesteps(total_steps, device=device)
    steps_to_run = int(total_steps * strength)
    start_idx = total_steps - steps_to_run
    
    t_start = scheduler.timesteps[start_idx]
    noise = torch.randn(latent_he.shape, generator=g, device=device, dtype=latent_he.dtype)
    t_start_batched = torch.full((B,), t_start, device=device, dtype=torch.long)
    xt = scheduler.add_noise(latent_he, noise, t_start_batched)
    timesteps = scheduler.timesteps[start_idx:]

    for i, t in enumerate(timesteps):
        curr_t = torch.full((B,), t, device=device, dtype=torch.long)
        
        with torch.autocast("cuda", dtype=autocast_dtype):
            eps = transformer(xt, encoder_hidden_states=uni_cond, timestep=curr_t, return_dict=False)[0][:, :16, :, :]
            if cfg_scale > 1.0:
                eps_u = transformer(xt, encoder_hidden_states=uncond, timestep=curr_t, return_dict=False)[0][:, :16, :, :]
                eps = eps_u + cfg_scale * (eps - eps_u)

        xt_prev = scheduler.step(eps, t, xt, return_dict=False)[0]

        progress = i / len(timesteps)
        
        if progress < progress_threshold:
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                t_next_batched = torch.full((B,), t_next, device=device, dtype=torch.long)
                noise_for_he = torch.randn(latent_he.shape, generator=g, device=device, dtype=latent_he.dtype)
                he_noisy = scheduler.add_noise(latent_he, noise_for_he, t_next_batched)
            else:
                he_noisy = latent_he 

            xt_prev = (mask_latent * he_noisy) + ((1.0 - mask_latent) * xt_prev)

        xt = xt_prev

    with torch.autocast("cuda", dtype=autocast_dtype):
        dec = vae.decode(xt / vae.config.scaling_factor, return_dict=False)[0]
    return (0.5 * (dec + 1)).clamp(0, 1)


def main():
    args = parse_args()
    out_dir = args.output_base_dir
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        set_perf_flags(args)

    autocast_dtype = torch.bfloat16 if (device.type == "cuda" and args.use_bf16) else torch.float16

    print("Caricamento Modelli (SDEdit Batch Mode)...")
    uni_model, uni_cfg = load_uni(device)
    sd3_vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        token=os.environ.get("HF_TOKEN"),
        subfolder="vae"
    ).to(device).eval()

    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "StonyBrook-CVLab/PixCell-1024",
        subfolder="scheduler"
    )

    transformer = build_transformer_with_lora(device, args.target, args.num_tokens)
    uni_mlp = build_flow_mlp(device, args.flow_target)
    stardist_model = StarDist2D.from_pretrained('2D_versatile_he')

    files = [f for f in os.listdir(args.input_dir) if f.endswith(".npz")]
    files.sort()

    if args.debug_file is not None:
        dbg = args.debug_file
        if os.path.isabs(dbg) or os.path.sep in dbg:
            input_path = dbg
            args.input_dir = os.path.dirname(input_path)
            files = [os.path.basename(input_path)]
        else:
            files = [dbg]

    for filename in files:
        print(f"\nElaborazione File: {filename}")
        input_path = os.path.join(args.input_dir, filename)
        output_filename = filename.replace("he_", "ihc_")
        out_path = os.path.join(out_dir, output_filename)

        mmap_mode = "r" if args.mmap_npz else None
        data = np.load(input_path, allow_pickle=True, mmap_mode=mmap_mode)
        arr = data[args.key]

        N_total = int(arr.shape[0])
        N = min(N_total, args.max_images_per_file)
        generated_data = []

        for start in tqdm(range(0, N, args.batch_size), desc="Batch Processing"):
            end = min(start + args.batch_size, N)
            
            batch_tiles_orig = []
            batch_labels = []
            batch_prec_imgs = []
            batch_mask_tensors = []

            for i in range(start, end):
                tile, lab = arr[i, 0], arr[i, 1]
                if isinstance(tile, np.ndarray) and tile.shape == (256, 256, 3):
                    tile = np.array(Image.fromarray(tile).resize((1024, 1024), Image.BICUBIC))
                if args.bgr:
                    tile = tile[..., ::-1].copy()

                batch_tiles_orig.append(tile.astype(np.uint8))
                batch_labels.append(lab)

                # Processamento Morfologico (StarDist + Colore)
                prec_img, mask_clean = process_tile_morphology(
                    tile, stardist_model, args.hsv_sat_scale, args.bg_h_preserve, args.stardist_prob_thresh
                )
                
                # Salviamo sia l'immagine che la maschera (Fix del bug precedente)
                batch_prec_imgs.append((prec_img, mask_clean))
                
                m_tensor = torch.from_numpy(mask_clean).unsqueeze(0).unsqueeze(0)
                m_latent = F.interpolate(m_tensor, size=(128, 128), mode='nearest').to(device)
                batch_mask_tensors.append(m_latent.repeat(1, 16, 1, 1))

            t0 = time.perf_counter()

            batch_np = np.stack(batch_tiles_orig, axis=0).astype(np.uint8)
            uni_emb_he = extract_uni_from_batch_tiles_fast(batch_np, uni_model, uni_cfg, device, autocast_dtype, args.num_tokens)
            uni_emb_ihc = flow_he_to_ihc(uni_emb_he, uni_mlp, args.flow_steps, autocast_dtype, args.num_tokens)
            
            norm_he = uni_emb_he.norm(dim=-1).mean(dim=1)
            norm_ihc = uni_emb_ihc.norm(dim=-1).mean(dim=1)
            scale = (norm_he / (norm_ihc + 1e-6)).detach().view(-1, 1, 1)
            uni_cond = uni_emb_ihc * scale

            # Codifica VAE delle immagini pre-condizionate
            prec_tensors = [item[0] for item in batch_prec_imgs]
            prec_tensor = (torch.from_numpy(np.stack(prec_tensors)).float().to(device) / 127.5) - 1.0
            prec_tensor = prec_tensor.permute(0, 3, 1, 2).contiguous()
            with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype):
                latent_he_batch = sd3_vae.encode(prec_tensor).latent_dist.sample() * sd3_vae.config.scaling_factor
            
            mask_latent_batch = torch.cat(batch_mask_tensors, dim=0)

            # --- PRIMO PASSAGGIO: Generazione Base ---
            # --- PRIMO PASSAGGIO: Generazione Base (Strength normale, es. 0.70) ---
            decoded = run_hybrid_sdedit_batch(
                transformer, sd3_vae, scheduler, uni_cond, latent_he_batch, mask_latent_batch,
                args.seed + start, args.num_inference_steps, args.strength, args.guidance_scale, 
                autocast_dtype, device, args.progress_threshold
            )

            decoded_np = (decoded.permute(0, 2, 3, 1).float().cpu().numpy() * 255.0).astype(np.uint8)
            
            for bi in range(decoded_np.shape[0]):
                gen_img = decoded_np[bi]
                mask_clean_uint8 = batch_prec_imgs[bi][1].astype(np.uint8)
                
                # 1. Troviamo le allucinazioni
                img_norm_gen = normalize(gen_img, 1, 99.8, axis=(0,1,2))
                labels_gen, _ = stardist_model.predict_instances(img_norm_gen, prob_thresh=args.stardist_prob_thresh)
                mask_gen = (labels_gen > 0).astype(np.uint8)
                
                false_nuclei = cv2.subtract(mask_gen, mask_clean_uint8)
                
                # Se l'immagine è già perfetta, salviamola e passiamo oltre!
                if cv2.countNonZero(false_nuclei) == 0:
                    generated_data.append([gen_img, batch_labels[bi]])
                    continue
                
                # 2. CHIRURGIA STRUTTURALE (OpenCV distrugge la geometria)
                kernel = np.ones((5,5), np.uint8)
                false_nuclei_dilated = cv2.dilate(false_nuclei, kernel, iterations=1)
                
                # OpenCV crea la "toppa" artificiale, distruggendo l'allucinazione
                patched_img = cv2.inpaint(gen_img, false_nuclei_dilated * 255, 3, cv2.INPAINT_TELEA)
                
                # 3. TEXTURE HARMONIZATION PASS (La Rete Neurale sistema l'estetica)
                # Rimettiamo l'immagine rattoppata nello spazio latente
                patched_tensor = (torch.from_numpy(patched_img).float().to(device) / 127.5) - 1.0
                patched_tensor = patched_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()
                
                with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype):
                    latent_patched = sd3_vae.encode(patched_tensor).latent_dist.sample() * sd3_vae.config.scaling_factor
                
                # Eseguiamo una diffusione leggerissima (Strength = 0.45) su TUTTA l'immagine.
                # Nota: teniamo mask_latent_batch attivo così i veri nuclei restano perfettamente intatti.
                harmonized_latent = run_hybrid_sdedit_batch(
                    transformer, sd3_vae, scheduler, uni_cond[bi:bi+1], latent_patched, mask_latent_batch[bi:bi+1],
                    args.seed + 999, args.num_inference_steps, 0.45, args.guidance_scale, 
                    autocast_dtype, device, 1.0
                )
                
                # 4. Decodifica e Salvataggio
                final_np = (harmonized_latent.permute(0, 2, 3, 1).float().cpu().numpy()[0] * 255.0).astype(np.uint8)
                generated_data.append([final_np, batch_labels[bi]])

            t1 = time.perf_counter()
            if args.profile:
                tqdm.write(f"Batch {start}:{end} completato in {t1 - t0:.2f}s")

        print(f"Salvataggio di {len(generated_data)} tiles in {out_path}...")
        np.savez(out_path, **{args.key: np.array(generated_data, dtype=object)})

if __name__ == "__main__":
    main()
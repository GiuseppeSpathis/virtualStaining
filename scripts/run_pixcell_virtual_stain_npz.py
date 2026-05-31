import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import einops
from PIL import Image
from tqdm.auto import tqdm

import timm
from timm import layers
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from huggingface_hub import hf_hub_download
from peft import LoraConfig

from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from virtual_staining.pixcell_transformer_2d_lora import PixCellTransformer2DModelLoRA
from virtual_staining.resmlp import SimpleMLP


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_images_per_file", type=int, default=300)
    p.add_argument("--input_dir", default=os.path.expanduser("~/giuSpathis/data/he"))
    p.add_argument("--output_base_dir", default=os.path.expanduser("~/giuSpathis/data/pixcellGenIhc"))
    p.add_argument("--key", default="arr_0", help="Key inside the npz")
    p.add_argument("--target", default="mist_pr",
                   help="Checkpoint name or path to a custom .pth LoRA file")
    p.add_argument("--flow_target", default=None)
    p.add_argument("--debug_file", type=str, default=None, help="If set, run inference only on this .npz file name (e.g. he_123.npz) or full path")
    p.add_argument("--bgr", action="store_true",
                   help="Interpret input tiles as BGR instead of RGB")
    p.add_argument("--no_flow", action="store_true",
                   help="Skip the Flow MLP step")
    p.add_argument("--device", default="cuda", help='e.g. "cuda", "cpu"')
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_inference_steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=1.2)
    p.add_argument("--flow_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--use_tf32", action="store_true")
    p.add_argument("--compile_transformer", action="store_true")
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
    uni_transform = create_transform(**uni_cfg)
    uni_model.eval().to(device)
    return uni_model, uni_transform, uni_cfg


def build_transformer_with_lora(device, target: str):
    config = {
        "_class_name": "PixCellTransformer2DModel", "_diffusers_version": "0.32.2",
        "_name_or_path": "pixart_1024/transformer", "activation_fn": "gelu-approximate",
        "attention_bias": True, "attention_head_dim": 72, "attention_type": "default",
        "caption_channels": 1536, "caption_num_tokens": 16, "cross_attention_dim": 1152,
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
    transformer.add_adapter(LoraConfig(
        r=4, lora_alpha=4, init_lora_weights="gaussian", target_modules=target_modules
    ))

    if os.path.exists(target):
        ckpt_path = target
    else:
        lora_map = {
            "mist_her2": "ckpts/lora/mist_her2_lora.pth",
            "mist_er": "ckpts/lora/mist_er_lora.pth",
            "mist_pr": "ckpts/lora/mist_pr_lora.pth",
            "mist_ki67": "ckpts/lora/mist_ki67_lora.pth",
            "her2match": "ckpts/lora/her2match_lora.pth",
        }
        ckpt_path = hf_hub_download(
            repo_id="StonyBrook-CVLab/pixcell-virtual-staining",
            filename=lora_map[target],
            local_dir="downloads/",
        )

    transformer.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    transformer.eval().to(device)
    return transformer


def build_flow_mlp(device, target: str, flow_target: str = None):
    use = flow_target if flow_target is not None else (None if os.path.exists(target) else target)
    if use is None:
        return None

    uni_mlp = SimpleMLP(
        in_channels=1536, time_embed_dim=1024, model_channels=1024,
        bottleneck_channels=1024, out_channels=1536, num_res_blocks=6
    ).to(device)

    if os.path.exists(use):
        ckpt_path = use
    else:
        mlp_map = {
            "mist_her2": "ckpts/mlp/mist_her2_mlp.pth",
            "mist_er": "ckpts/mlp/mist_er_mlp.pth",
            "mist_pr": "ckpts/mlp/mist_pr_mlp.pth",
            "mist_ki67": "ckpts/mlp/mist_ki67_mlp.pth",
            "her2match": "ckpts/mlp/her2match_mlp.pth",
        }
        ckpt_path = hf_hub_download(
            repo_id="StonyBrook-CVLab/pixcell-virtual-staining",
            filename=mlp_map[use],
            local_dir="downloads/",
        )

    uni_mlp.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    uni_mlp.eval().to(device)
    return uni_mlp



@torch.no_grad()
def extract_uni_from_batch_tiles_fast(batch_tiles_uint8, uni_model, uni_cfg, device, autocast_dtype):
    
    x = torch.from_numpy(batch_tiles_uint8).to(device, non_blocking=True)  # (B,H,W,C) uint8
    x = x.float().div_(255.0)
    x = x.permute(0, 3, 1, 2).contiguous()  # (B,3,1024,1024)

    # split 1024 -> 16 patch 256
    patches = x.unfold(2, 256, 256).unfold(3, 256, 256)  # (B,3,4,4,256,256)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, 3, 256, 256)  # (B*16,3,256,256)

    input_size = uni_cfg["input_size"]  
    out_h, out_w = input_size[-2], input_size[-1]
    patches = F.interpolate(patches, size=(out_h, out_w), mode="bicubic", align_corners=False)

    mean = torch.tensor(uni_cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(uni_cfg["std"], device=device).view(1, 3, 1, 1)
    patches = (patches - mean) / std

    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")):
        emb = uni_model(patches)  # (B*16,1536)

    B = batch_tiles_uint8.shape[0]
    return emb.view(B, 16, -1)


@torch.no_grad()
def extract_uni_from_tile_slow(tile_uint8, uni_model, uni_transform, device, autocast_dtype):
   
    patches = einops.rearrange(tile_uint8, "(d1 h) (d2 w) c -> (d1 d2) h w c", d1=4, d2=4)
    uni_tensors = [uni_transform(Image.fromarray(p)) for p in patches]
    uni_inp = torch.stack(uni_tensors).to(device, non_blocking=True)
    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")):
        emb = uni_model(uni_inp)
    return emb.unsqueeze(0)


@torch.no_grad()
def flow_he_to_ihc(uni_emb_he, uni_mlp, flow_steps, autocast_dtype):
    
    B = uni_emb_he.shape[0]
    x = uni_emb_he.reshape(-1, 1536).float()

    steps = int(max(flow_steps, 1))
    dt = 1.0 / steps
    ts = torch.linspace(1e-3, 1.0 - 1e-3, steps=steps, device=x.device, dtype=torch.float32)

    base = torch.empty((B * 16,), device=x.device, dtype=torch.float32)
    for t in ts:
        base.fill_(t)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(x.is_cuda)):
            dx = uni_mlp(x, 999.0 * base)
        x = x + dt * dx.float()

    return x.view(B, 16, 1536)


@torch.no_grad()
def sample_image_from_uni(transformer, sd3_vae, scheduler, uni_cond, steps, guidance, seed, autocast_dtype):
    device = uni_cond.device
    B = uni_cond.shape[0]
    uncond = transformer.caption_projection.uncond_embedding.clone().tile(B, 1, 1).to(device)

    vae_scale = sd3_vae.config.scaling_factor
    vae_shift = getattr(sd3_vae.config, "shift_factor", 0)

    g = torch.Generator(device=device).manual_seed(seed)
    xt = torch.randn((B, 16, 128, 128), device=device, generator=g)

    scheduler.set_timesteps(steps, device=device)

    for tt in tqdm(scheduler.timesteps, desc="Diffusion", leave=False):
        curr_t = torch.full((B,), tt, device=device, dtype=torch.long)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")):
            eps = transformer(xt, encoder_hidden_states=uni_cond, timestep=curr_t, return_dict=False)[0][:, :16, :, :]
            if guidance > 1.0:
                eps_un = transformer(xt, encoder_hidden_states=uncond, timestep=curr_t, return_dict=False)[0][:, :16, :, :]
                eps = eps_un + guidance * (eps - eps_un)
        xt = scheduler.step(eps, tt, xt, return_dict=False)[0]

    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")):
        decoded = sd3_vae.decode((xt / vae_scale) + vae_shift, return_dict=False)[0]

    return (0.5 * (decoded + 1)).clamp(0, 1)


def main():
    args = parse_args()

    checkpoint_name = os.path.basename(args.target).replace(".pth", "")
    out_dir = os.path.join(args.output_base_dir, checkpoint_name)
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        set_perf_flags(args)

    autocast_dtype = torch.bfloat16 if (device.type == "cuda" and args.use_bf16) else torch.float16

    # Models
    uni_model, uni_transform, uni_cfg = load_uni(device)
    sd3_vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large",
        token=os.environ.get("HF_TOKEN"),
        subfolder="vae"
    ).to(device).eval()

    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "StonyBrook-CVLab/PixCell-1024",
        subfolder="scheduler"
    )

    transformer = build_transformer_with_lora(device, args.target)
    if device.type == "cuda" and args.compile_transformer:
        transformer = torch.compile(transformer, mode="reduce-overhead")
        
    uni_mlp = build_flow_mlp(device, args.target, args.flow_target)

    files = [f for f in os.listdir(args.input_dir) if f.endswith(".npz")]
    files.sort()

    if args.debug_file is not None:
        dbg = args.debug_file

        # se passi un path completo
        if os.path.isabs(dbg) or os.path.sep in dbg:
            input_path = dbg
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"--debug_file path not found: {input_path}")
            if not input_path.endswith(".npz"):
                raise ValueError(f"--debug_file must point to a .npz file: {input_path}")

            args.input_dir = os.path.dirname(input_path)
            files = [os.path.basename(input_path)]
        else:
            # se passi solo il nome file, es. he_1811474_3.npz
            if dbg not in files:
                raise FileNotFoundError(f"--debug_file '{dbg}' not found in input_dir={args.input_dir}. "f"Available files: {files[:10]}{'...' if len(files) > 10 else ''}")
            files = [dbg]

        print(f"[DEBUG] Running inference only on file: {files[0]}")

    for filename in files:
        input_path = os.path.join(args.input_dir, filename)
        output_filename = filename.replace("he_", "ihc_")
        out_path = os.path.join(out_dir, output_filename)

        
        mmap_mode = "r" if args.mmap_npz else None
        data = np.load(input_path, allow_pickle=True, mmap_mode=mmap_mode)
        arr = data[args.key]

        N_total = int(arr.shape[0])
        N = min(N_total, args.max_images_per_file)

        generated_data = []

        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            batch_tiles = []
            batch_labels = []

            for i in range(start, end):
                tile, lab = arr[i, 0], arr[i, 1]

                if isinstance(tile, np.ndarray) and tile.shape == (256, 256, 3):
                    tile = np.array(Image.fromarray(tile).resize((1024, 1024), Image.BICUBIC))

                if args.bgr:
                    tile = tile[..., ::-1].copy()

                batch_tiles.append(tile.astype(np.uint8))
                batch_labels.append(lab)

            t0 = time.perf_counter()

            batch_np = np.stack(batch_tiles, axis=0).astype(np.uint8)
            uni_emb_he = extract_uni_from_batch_tiles_fast(batch_np, uni_model, uni_cfg, device, autocast_dtype)

            t1 = time.perf_counter()

            if args.no_flow or uni_mlp is None:
                uni_emb_ihc = uni_emb_he
            else:
                uni_emb_ihc = flow_he_to_ihc(uni_emb_he, uni_mlp, args.flow_steps, autocast_dtype)

            t2 = time.perf_counter()

            scale = (uni_emb_he.norm(dim=-1).mean() / (uni_emb_ihc.norm(dim=-1).mean() + 1e-6)).detach()
            uni_emb_ihc = uni_emb_ihc * scale

            decoded = sample_image_from_uni(
                transformer, sd3_vae, scheduler,
                uni_emb_ihc,
                args.num_inference_steps,
                args.guidance_scale,
                args.seed + start,
                autocast_dtype
            )

            t3 = time.perf_counter()

            decoded_np = (decoded.permute(0, 2, 3, 1).float().cpu().numpy() * 255.0).astype(np.uint8)
            for bi in range(decoded_np.shape[0]):
                generated_data.append([decoded_np[bi], batch_labels[bi]])

            if args.profile:
                print(
                    f"[batch {start}:{end}] UNI {t1 - t0:.2f}s | "
                    f"flow {t2 - t1:.2f}s | diffusion+decode {t3 - t2:.2f}s | total {t3 - t0:.2f}s"
                )

        output_filename = filename.replace("he_", "ihc_")
        out_path = os.path.join(out_dir, output_filename)
        np.savez(out_path, **{args.key: np.array(generated_data, dtype=object)})


if __name__ == "__main__":
    main()


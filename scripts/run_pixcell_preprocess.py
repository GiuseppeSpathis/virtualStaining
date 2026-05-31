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

from peft import LoraConfig
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler
from virtual_staining.pixcell_transformer_2d_lora import PixCellTransformer2DModelLoRA
from virtual_staining.resmlp import SimpleMLP
from stardist.models import StarDist2D
from csbdeep.utils import normalize


def parse_args():
    """
    Parses command-line arguments for the virtual staining generation pipeline.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--max_images_per_file", type=int, default=300)
    p.add_argument("--input_dir", default=os.path.expanduser("~/giuSpathis/data/he"))
    p.add_argument("--output_base_dir", default=os.path.expanduser("~/giuSpathis/data/pixcellGenIhc_consensus"))
    p.add_argument("--key", default="arr_0", help="Key inside the npz")
    p.add_argument("--target", required=True, help="Path to a custom .pth LoRA file")
    p.add_argument("--flow_target", required=True)
    p.add_argument("--debug_file", type=str, default=None, help="If set, run inference only on this .npz file name")
    p.add_argument("--bgr", action="store_true", help="Interpret input tiles as BGR instead of RGB")
    p.add_argument("--device", default="cuda", help='e.g. "cuda", "cpu"')
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--parallel_seeds", type=int, default=1, help="Number of inferences (seeds) to process simultaneously in VRAM.")
    p.add_argument("--num_tokens", type=int, default=16)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=1.2)
    p.add_argument("--flow_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--strength", type=float, default=0.55, help="SDEdit noise strength")
    p.add_argument("--progress_threshold", type=float, default=0.65, help="Percentage of steps for the lock")
    p.add_argument("--hsv_sat_scale", type=float, default=0.4, help="Background desaturation scale")
    p.add_argument("--bg_h_preserve", type=float, default=0.2, help="Hematoxylin background preservation")
    p.add_argument("--stardist_prob_thresh", type=float, default=0.15, help="StarDist probability threshold")
    p.add_argument("--consensus_n_seeds", type=int, default=5, help="Number of random seeds per tile")
    p.add_argument("--consensus_seed_list", type=str, default=None)
    p.add_argument("--save_seed_outputs", action="store_true", help="Save individual seed outputs in subfolders")
    p.add_argument("--dab_vote_min", type=int, default=3, help="Minimum votes out of N seeds to accept DAB consensus")
    p.add_argument("--dab_abs_threshold", type=float, default=0.035)
    p.add_argument("--dab_percentile", type=float, default=70.0)
    p.add_argument("--consensus_dab_alpha", type=float, default=0.75)
    p.add_argument("--outside_dab_suppress", type=float, default=0.35)
    p.add_argument("--consensus_mask_blur", type=int, default=9)
    p.add_argument("--cell_dilate", type=int, default=23)
    p.add_argument("--skip_consensus", action="store_true", help="Disable consensus and save the first seed")
    p.add_argument("--disable_false_nuclei_inpaint", action="store_true", help="Disable inpainting for false nuclei")
    p.add_argument("--harmonization_strength", type=float, default=0.35)
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--use_tf32", action="store_true")
    p.add_argument("--mmap_npz", action="store_true", default=True)
    p.add_argument("--empty_cuda_cache", action="store_true")
    p.add_argument("--tile_idx", type=int, default=None)
    p.add_argument("--randomize_consensus_seeds", action="store_true")
    p.add_argument("--reference_ihc", type=str, default=None, help="Path to real IHC .png/.jpg tile for Reinhard reference")
    p.add_argument("--reinhard_color_blur", type=int, default=5, help="Blur strength for Reinhard color.")
    
    return p.parse_args()


def get_nuclei_lab_stats(rgb_uint8, mask):
    """
    Extracts the mean and standard deviation of LAB color channels for the masked nuclei regions.
    """
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    m = mask > 0.5
    
    if m.sum() == 0:
        return np.array([135.0, 128.0, 110.0]), np.array([15.0, 5.0, 8.0])
        
    means = np.array([np.mean(lab[:, :, i][m]) for i in range(3)])
    stds = np.array([np.std(lab[:, :, i][m]) + 1e-6 for i in range(3)])
    return means, stds


def apply_reinhard_nuclei(rgb_uint8, mask, target_means, target_stds, color_blur_ksize=3):
    """
    Applies Reinhard color normalization exclusively to the nuclei regions and smoothly blends them with the background.
    """
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    m = mask > 0.5

    if m.sum() < 20:
        return rgb_uint8

    for i in range(3):
        c = lab[:, :, i]
        src_mean = np.mean(c[m])
        src_std = np.std(c[m]) + 1e-6
        c[m] = ((c[m] - src_mean) / src_std) * target_stds[i] + target_means[i]
        lab[:, :, i] = c

    pure_reinhard_rgb = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)

    if color_blur_ksize % 2 == 0:
        color_blur_ksize += 1

    if color_blur_ksize > 1:
        reinhard_blurred = cv2.GaussianBlur(pure_reinhard_rgb, (color_blur_ksize, color_blur_ksize), 0)
    else:
        reinhard_blurred = pure_reinhard_rgb

    mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (3, 3), 0)
    mask_3d = np.expand_dims(mask_blur, axis=-1)

    final_out = (mask_3d * reinhard_blurred.astype(np.float32)) + ((1.0 - mask_3d) * rgb_uint8.astype(np.float32))
    
    return np.clip(final_out, 0, 255).astype(np.uint8)


def set_perf_flags(args):
    """
    Configures PyTorch CUDA backend settings to optimize matrix multiplication precision and speed.
    """
    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    torch.backends.cudnn.benchmark = True


def load_uni(device):
    """
    Initializes and loads the pre-trained UNI vision transformer model and its data configuration.
    """
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
    """
    Instantiates the PixCellTransformer2DModel and injects customized LoRA weights for generation.
    """
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
    """
    Loads the multi-layer perceptron model responsible for translating embeddings via flow matching.
    """
    uni_mlp = SimpleMLP(
        in_channels=1536, time_embed_dim=1024, model_channels=1024,
        bottleneck_channels=1024, out_channels=1536, num_res_blocks=6
    ).to(device)
    uni_mlp.load_state_dict(torch.load(flow_target, map_location="cpu"))
    uni_mlp.eval().to(device)
    return uni_mlp


@torch.no_grad()
def extract_uni_from_batch_tiles_fast(batch_tiles_uint8, uni_model, uni_cfg, device, autocast_dtype, num_tokens=16):
    """
    Extracts high-dimensional patch embeddings from a batch of input image tiles using the UNI model.
    """
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
    """
    Simulates a flow matching process to translate the extracted H&E embeddings into targeted IHC embeddings.
    """
    B = uni_emb_he.shape[0]
    x = uni_emb_he.reshape(-1, 1536).float()
    dt = 1.0 / flow_steps
    ts = torch.linspace(1e-3, 1.0 - 1e-3, steps=flow_steps, device=x.device, dtype=torch.float32)
    for t in ts:
        time_tensor = torch.full((num_tokens * B,), t, device=x.device) * 999.0
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=x.is_cuda):
            dx = uni_mlp(x, time_tensor)
        x = x + dt * dx.float()
    return x.view(B, num_tokens, 1536)


def process_tile_morphology(he_np, stardist_model, hsv_sat_scale, bg_h_preserve, prob_thresh, args):
    """
    Analyzes H&E tile morphology with StarDist to extract cleaned nuclei instance masks, filtering outliers.
    """
    img_norm = normalize(he_np, 1, 99.8, axis=(0, 1, 2))
    labels, _ = stardist_model.predict_instances(img_norm, prob_thresh=prob_thresh)

    mask_clean = np.zeros_like(labels, dtype=np.float32)
    for prop in regionprops(labels):
        if prop.area < 40 or prop.area > 2000 or prop.eccentricity > 0.85:
            continue
        mask_clean[labels == prop.label] = 1.0

    kernel_dilate = np.ones((3, 3), np.uint8)
    mask_clean = cv2.dilate(mask_clean, kernel_dilate, iterations=1).astype(np.float32)

    return he_np.astype(np.uint8), mask_clean


def encode_latent_mode(vae, tensor_bchw_minus1_1, autocast_dtype, device):
    """
    Encodes an image tensor into the latent space using the variational autoencoder's distribution mode.
    """
    with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")):
        posterior = vae.encode(tensor_bchw_minus1_1).latent_dist
        if hasattr(posterior, "mode"):
            latent = posterior.mode()
        else:
            latent = posterior.mean
        return latent * vae.config.scaling_factor


@torch.no_grad()
def run_hybrid_sdedit_batch(transformer, vae, scheduler, uni_cond, latent_he, mask_latent,
                            seeds, total_steps, strength, cfg_scale, autocast_dtype, device, progress_threshold):
    """
    Executes a batched SDEdit diffusion process ensuring exact reproducibility via isolated random generators for each seed.
    """
    B = uni_cond.shape[0]
    uncond = transformer.caption_projection.uncond_embedding.clone().tile(B, 1, 1).to(device)
    
    generators = [torch.Generator(device=device).manual_seed(int(s)) for s in seeds]

    def gen_exact_noise():
        noises = []
        for g in generators:
            noises.append(torch.randn((1, *latent_he.shape[1:]), generator=g, device=device, dtype=latent_he.dtype))
        return torch.cat(noises, dim=0)

    scheduler.set_timesteps(total_steps, device=device)
    steps_to_run = max(1, min(int(total_steps * strength), total_steps))
    start_idx = total_steps - steps_to_run
    
    lock_steps = int(steps_to_run * progress_threshold)

    t_start = scheduler.timesteps[start_idx]
    t_start_batched = torch.full((B,), t_start, device=device, dtype=torch.long)
    
    noise = gen_exact_noise()
    xt = scheduler.add_noise(latent_he, noise, t_start_batched)
    
    timesteps = scheduler.timesteps[start_idx:]
    soft_mask = mask_latent

    for i, t in enumerate(timesteps):
        curr_t = torch.full((B,), t, device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")):
            eps = transformer(xt, encoder_hidden_states=uni_cond, timestep=curr_t, return_dict=False)[0][:, :16, :, :]
            if cfg_scale > 1.0:
                eps_u = transformer(xt, encoder_hidden_states=uncond, timestep=curr_t, return_dict=False)[0][:, :16, :, :]
                eps = eps_u + cfg_scale * (eps - eps_u)

        xt_prev = scheduler.step(eps, t, xt, return_dict=False)[0]

        if i < lock_steps:
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                t_next_batched = torch.full((B,), t_next, device=device, dtype=torch.long)
                noise_for_he = gen_exact_noise()
                he_noisy = scheduler.add_noise(latent_he, noise_for_he, t_next_batched)
            else:
                he_noisy = latent_he
            
            xt_prev = (soft_mask * he_noisy) + ((1.0 - soft_mask) * xt_prev)

        xt = xt_prev

    with torch.autocast("cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")):
        dec = vae.decode(xt / vae.config.scaling_factor, return_dict=False)[0]
    return (0.5 * (dec + 1)).clamp(0, 1)


def tissue_mask_rgb(rgb, s_thresh=0.04, v_thresh=0.94):
    """
    Computes a binary tissue map isolating background areas using HSV saturation and value channels.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0
    tissue = ((s > s_thresh) & (v < v_thresh)).astype(np.float32)
    return cv2.GaussianBlur(tissue, (9, 9), 0).clip(0, 1)


def allowed_dab_mask_from_he(he_rgb, nuclei_mask, cell_dilate):
    """
    Calculates the allowed spatial region for DAB stain precipitation based on cell vicinity and general tissue presence.
    """
    tissue = tissue_mask_rgb(he_rgb)
    k = max(3, int(cell_dilate))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), np.uint8)
    cell = cv2.dilate((nuclei_mask > 0.1).astype(np.uint8), kernel, iterations=1).astype(np.float32)
    cell = cv2.GaussianBlur(cell, (9, 9), 0).clip(0, 1)
    allowed = np.maximum(0.35 * tissue, cell * tissue)
    return allowed.clip(0, 1)


def hed_dab_channel(rgb_uint8):
    """
    Converts the RGB matrix to HED color space and isolates the DAB optical density channel.
    """
    rgb01 = rgb_uint8.astype(np.float32) / 255.0
    hed = rgb2hed(rgb01)
    d = hed[:, :, 2].astype(np.float32)
    return hed, d


def select_best_seed_image(images, allowed_mask):
    """
    Scores and selects the best generated outcome among multiple seeds using DAB statistics, saturation, and contrast.
    """
    scores = []
    allowed = allowed_mask > 0.05
    if allowed.sum() < 100:
        allowed = np.ones(allowed_mask.shape, dtype=bool)
    for im in images:
        hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = hsv[:, :, 1] / 255.0
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F).var()
        _, d = hed_dab_channel(im)
        d_allowed = d[allowed]
        dab_mean = float(np.mean(np.clip(d_allowed, 0, None)))
        dab_frac = float(np.mean(d_allowed > np.percentile(d_allowed, 70))) if d_allowed.size else 0.0
        imf = im.astype(np.float32) / 255.0
        green_cast = float(np.mean(np.maximum(0.0, imf[:, :, 1] - 0.5 * (imf[:, :, 0] + imf[:, :, 2]))))
        sat_std = float(np.std(sat[allowed])) if allowed.any() else float(np.std(sat))
        score = 2.0 * dab_mean + 0.3 * dab_frac + 0.15 * sat_std + 0.00005 * lap - 0.8 * green_cast
        scores.append(score)
    return int(np.argmax(scores)), scores


def consensus_stain_aware(images, he_rgb, nuclei_mask, args):
    """
    Aggregates multi-seed generated images to evaluate structural consistency and selects the generation matching the consensus DAB map.
    """
    if len(images) == 1 or args.skip_consensus:
        return images[0]

    n = len(images)
    vote_min = min(max(1, args.dab_vote_min), n)

    allowed = allowed_dab_mask_from_he(he_rgb, nuclei_mask, args.cell_dilate)
    allowed_bool = allowed > 0.03

    d_maps = []
    votes = []

    for im in images:
        _, d = hed_dab_channel(im)
        d_maps.append(d)

        valid = d[allowed_bool] if allowed_bool.any() else d.reshape(-1)
        thr = max(
            float(args.dab_abs_threshold),
            float(np.percentile(valid, args.dab_percentile))
        )

        votes.append((d > thr) & allowed_bool)

    d_stack = np.stack(d_maps, axis=0)
    votes_stack = np.stack(votes, axis=0).astype(np.uint8)

    consensus_mask = votes_stack.sum(axis=0) >= vote_min

    if consensus_mask.sum() < 50:
        base_idx, _ = select_best_seed_image(images, allowed)
        return images[base_idx]

    median_dab = np.median(d_stack, axis=0)

    scores = []
    for idx, d in enumerate(d_maps):
        diff_consensus = np.mean(np.abs(d[consensus_mask] - median_dab[consensus_mask]))
        
        outside = allowed_bool & (~consensus_mask)
        
        if outside.sum() > 50:
            outside_penalty = np.mean(np.clip(d[outside], 0, None))
            penalty_weight = 0.5
        else:
            outside_penalty = 0.0

        dab_inside = np.mean(np.clip(d[consensus_mask], 0, None))
        
        score = -diff_consensus - (penalty_weight * outside_penalty) + 0.2 * dab_inside
        scores.append(score)

    best_idx = int(np.argmax(scores))
    return images[best_idx]


def inpaint_and_harmonize_one(gen_img, batch_label, mask_clean, stardist_model, prob_thresh,
                              transformer, vae, scheduler, uni_cond_one, latent_mask_one,
                              seed, args, autocast_dtype, device):
    """
    Detects hallucinated false nuclei in the synthesized output, masks them, and runs an inpainting harmonization pass.
    """
    if getattr(args, "disable_false_nuclei_inpaint", False):
        return gen_img

    img_norm_gen = normalize(gen_img, 1, 99.8, axis=(0, 1, 2))
    labels_gen, _ = stardist_model.predict_instances(img_norm_gen, prob_thresh=prob_thresh)
    mask_gen = (labels_gen > 0).astype(np.uint8)
    
    mask_clean_uint8 = (mask_clean > 0.1).astype(np.uint8)

    kernel_tolerance = np.ones((3, 3), np.uint8) 
    mask_clean_tolerant = cv2.dilate(mask_clean_uint8, kernel_tolerance, iterations=1)

    false_nuclei = cv2.subtract(mask_gen, mask_clean_tolerant)
    
    if cv2.countNonZero(false_nuclei) == 0:
        patched_img = gen_img
    else:
        kernel = np.ones((5, 5), np.uint8)
        false_nuclei_dilated = cv2.dilate(false_nuclei, kernel, iterations=1)
        
        patched_img = gen_img.copy()
        patched_img[false_nuclei_dilated > 0] = [240, 240, 245] 
        blurred_patch = cv2.GaussianBlur(patched_img, (9, 9), 0)
        
        mask_3d = np.expand_dims(false_nuclei_dilated, axis=-1)
        patched_img = np.where(mask_3d > 0, blurred_patch, patched_img).astype(np.uint8)

    if args.harmonization_strength <= 0:
        return patched_img

    patched_tensor = (torch.from_numpy(patched_img).float().to(device) / 127.5) - 1.0
    patched_tensor = patched_tensor.permute(2, 0, 1).unsqueeze(0).contiguous()

    with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype):
        latent_patched = vae.encode(patched_tensor).latent_dist.sample() * vae.config.scaling_factor

    harmonized = run_hybrid_sdedit_batch(
        transformer, vae, scheduler, uni_cond_one, latent_patched, latent_mask_one,
        [seed], args.num_inference_steps, args.harmonization_strength, args.guidance_scale,
        autocast_dtype, device, 1.0 
    )
    
    harmonized_img = (harmonized.permute(0, 2, 3, 1).float().cpu().numpy()[0] * 255.0).clip(0, 255).astype(np.uint8)

    return harmonized_img


def make_seed_list(args):
    """
    Parses a direct string or mathematically initializes the random generation seeds ensuring reproducibility.
    """
    if args.consensus_seed_list:
        seeds = [int(x.strip()) for x in args.consensus_seed_list.split(",") if x.strip()]
        if len(seeds) == 0:
            raise ValueError("--consensus_seed_list is empty or invalid.")
        return seeds

    n = int(args.consensus_n_seeds)
    if n <= 0:
        raise ValueError("--consensus_n_seeds must be > 0.")

    if getattr(args, "randomize_consensus_seeds", False):
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(args.seed)

    seeds = rng.integers(
        low=0,
        high=np.iinfo(np.int32).max,
        size=n,
        dtype=np.int64
    )
    return [int(s) for s in seeds]


def main():
    """
    Coordinates the main data processing loop mapping pre-processed H&E array batches to flow translations, diffusion outputs, and final saves.
    """
    args = parse_args()

    out_dir = args.output_base_dir
    if not out_dir:
        raise ValueError("--output_base_dir is empty: set OUTPUT_BASE in your bash script.")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        set_perf_flags(args)

    autocast_dtype = torch.bfloat16 if (device.type == "cuda" and args.use_bf16) else torch.float16

    print("Loading Models (SDEdit + harmonization + consensus)...")
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
    stardist_model = StarDist2D.from_pretrained("2D_versatile_he")

    target_means, target_stds = np.array([135.0, 128.0, 110.0]), np.array([15.0, 5.0, 8.0]) 
    if args.reference_ihc and os.path.exists(args.reference_ihc):
        print(f"Extracting Reinhard statistics from reference image: {args.reference_ihc}")
        ref_img = cv2.imread(args.reference_ihc)
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        
        ref_norm = normalize(ref_img, 1, 99.8, axis=(0, 1, 2))
        labels_ref, _ = stardist_model.predict_instances(ref_norm, prob_thresh=args.stardist_prob_thresh)
        ref_mask = (labels_ref > 0).astype(np.uint8)
        target_means, target_stds = get_nuclei_lab_stats(ref_img, ref_mask)
        print(f"Target Nuclei Statistics: LAB Mean {target_means}, LAB Std {target_stds}")
    elif args.reference_ihc:
        print(f"WARNING: The reference image {args.reference_ihc} does not exist. Using default values.")

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

    global_seed_list = make_seed_list(args)
    print(f"Consensus seeds: {global_seed_list}")

    for file_idx, filename in enumerate(files):
        print(f"\nProcessing File: {filename}")

        input_path = os.path.join(args.input_dir, filename)

        output_filename = filename.replace("he_", "ihc_")
        if args.tile_idx is not None:
            base, ext = os.path.splitext(output_filename)
            output_filename = f"{base}_tile{args.tile_idx}{ext}"

        out_path = os.path.join(out_dir, output_filename)

        mmap_mode = "r" if args.mmap_npz else None
        data = np.load(input_path, allow_pickle=True, mmap_mode=mmap_mode)
        arr = data[args.key]

        N_total = int(arr.shape[0])

        if args.tile_idx is not None:
            if args.tile_idx < 0 or args.tile_idx >= N_total:
                raise ValueError(
                    f"--tile_idx {args.tile_idx} out of range. "
                    f"File contains {N_total} tiles, valid indices: 0..{N_total - 1}"
                )
            tile_indices = [args.tile_idx]
        else:
            N = min(N_total, args.max_images_per_file)
            tile_indices = list(range(N))

        print(f"Total tiles in file: {N_total}")
        print(f"Tiles to process: {tile_indices[:10]}{'...' if len(tile_indices) > 10 else ''}")

        generated_data = []

        if args.save_seed_outputs:
            for s in global_seed_list:
                os.makedirs(os.path.join(out_dir, f"seed_{s}"), exist_ok=True)

        for start in tqdm(
            range(0, len(tile_indices), args.batch_size),
            desc="Batch Processing (Tiles)"
        ):
            end = min(start + args.batch_size, len(tile_indices))
            batch_tile_indices = tile_indices[start:end]

            batch_tiles_orig = []
            batch_labels = []
            batch_prec_imgs = []
            batch_mask_tensors = []
            batch_original_indices = []

            for i in batch_tile_indices:
                tile, lab = arr[i, 0], arr[i, 1]

                if isinstance(tile, np.ndarray) and tile.shape == (256, 256, 3):
                    tile = np.array(
                        Image.fromarray(tile.astype(np.uint8)).resize((1024, 1024), Image.BICUBIC)
                    )

                if args.bgr:
                    tile = tile[..., ::-1].copy()

                tile = tile.astype(np.uint8)

                batch_tiles_orig.append(tile)
                batch_labels.append(lab)
                batch_original_indices.append(i)

                prec_img, mask_clean = process_tile_morphology(
                    tile,
                    stardist_model,
                    args.hsv_sat_scale,
                    args.bg_h_preserve,
                    args.stardist_prob_thresh,
                    args
                )

                batch_prec_imgs.append((prec_img, mask_clean))

                m_tensor = torch.from_numpy(mask_clean).unsqueeze(0).unsqueeze(0)
                m_latent = F.interpolate(
                    m_tensor,
                    size=(128, 128),
                    mode="bilinear",
                    align_corners=False
                ).clamp(0.0, 1.0).to(device)

                batch_mask_tensors.append(m_latent.repeat(1, 16, 1, 1))

            batch_np = np.stack(batch_tiles_orig, axis=0).astype(np.uint8)

            uni_emb_he = extract_uni_from_batch_tiles_fast(
                batch_np,
                uni_model,
                uni_cfg,
                device,
                autocast_dtype,
                args.num_tokens
            )

            uni_emb_ihc = flow_he_to_ihc(
                uni_emb_he,
                uni_mlp,
                args.flow_steps,
                autocast_dtype,
                args.num_tokens
            )

            norm_he = uni_emb_he.norm(dim=-1).mean(dim=1)
            norm_ihc = uni_emb_ihc.norm(dim=-1).mean(dim=1)
            scale = (norm_he / (norm_ihc + 1e-6)).detach().view(-1, 1, 1)
            uni_cond = uni_emb_ihc * scale

            batch_for_vae = batch_np.copy()
            for bi in range(batch_for_vae.shape[0]):
                batch_for_vae[bi] = apply_reinhard_nuclei(
                    batch_for_vae[bi],
                    batch_prec_imgs[bi][1],
                    target_means,
                    target_stds,
                    color_blur_ksize=args.reinhard_color_blur
                )

            raw_tensor = (torch.from_numpy(batch_for_vae).float().to(device) / 127.5) - 1.0
            raw_tensor = raw_tensor.permute(0, 3, 1, 2).contiguous()

            latent_he_batch = encode_latent_mode(
                sd3_vae,
                raw_tensor,
                autocast_dtype,
                device
            )

            mask_latent_batch = torch.cat(batch_mask_tensors, dim=0)

            raw_seed_outputs_per_bi = [[] for _ in range(len(batch_tile_indices))]

            parallel_k = args.parallel_seeds
            seed_chunks = [global_seed_list[i:i + parallel_k] for i in range(0, len(global_seed_list), parallel_k)]

            for seed_chunk in seed_chunks:
                K = len(seed_chunk)
                
                expanded_uni_cond = uni_cond.repeat_interleave(K, dim=0)
                expanded_latent = latent_he_batch.repeat_interleave(K, dim=0)
                expanded_mask = mask_latent_batch.repeat_interleave(K, dim=0)
                
                run_seeds = []
                for bi in range(len(batch_tile_indices)):
                    for s in seed_chunk:
                        run_seeds.append(int(s) + int(batch_original_indices[bi]) * 1009)

                decoded = run_hybrid_sdedit_batch(
                    transformer,
                    sd3_vae,
                    scheduler,
                    expanded_uni_cond,
                    expanded_latent,
                    expanded_mask,
                    run_seeds,
                    args.num_inference_steps,
                    args.strength,
                    args.guidance_scale,
                    autocast_dtype,
                    device,
                    args.progress_threshold
                )

                decoded_np = (
                    decoded.permute(0, 2, 3, 1)
                    .float()
                    .cpu()
                    .numpy() * 255.0
                ).clip(0, 255).astype(np.uint8)

                idx = 0
                for bi in range(len(batch_tile_indices)):
                    for s_idx, s in enumerate(seed_chunk):
                        seed_img = decoded_np[idx]
                        raw_seed_outputs_per_bi[bi].append(seed_img)

                        if args.save_seed_outputs:
                            seed_out_dir = os.path.join(out_dir, f"seed_{s}")
                            seed_output_filename = output_filename
                            if args.tile_idx is not None or args.debug_file is not None:
                                png_base, _ = os.path.splitext(seed_output_filename)
                                png_name = f"{png_base}_idx{batch_original_indices[bi]}_seed{s}.png"
                                Image.fromarray(seed_img).save(os.path.join(seed_out_dir, png_name))
                        
                        idx += 1

            if args.empty_cuda_cache and device.type == "cuda":
                torch.cuda.empty_cache()

            for bi, raw_imgs in enumerate(raw_seed_outputs_per_bi):
                
                if args.skip_consensus:
                    best_raw_img = raw_imgs[0]
                else:
                    best_raw_img = consensus_stain_aware(
                        raw_imgs,
                        batch_tiles_orig[bi],
                        batch_prec_imgs[bi][1],
                        args
                    )

                final_img = inpaint_and_harmonize_one(
                    best_raw_img,
                    batch_labels[bi],
                    batch_prec_imgs[bi][1],
                    stardist_model,
                    args.stardist_prob_thresh,
                    transformer,
                    sd3_vae,
                    scheduler,
                    uni_cond[bi:bi + 1],
                    mask_latent_batch[bi:bi + 1],
                    args.seed + 9999,
                    args,
                    autocast_dtype,
                    device,
                )

                generated_data.append([
                    final_img,
                    batch_labels[bi],
                    batch_original_indices[bi]
                ])

        print(f"Saving {len(generated_data)} consensus tiles to {out_path}...")
        np.savez(out_path, **{args.key: np.array(generated_data, dtype=object)})

    print("Consensus complete.")
    
if __name__ == "__main__":
    main()
import os
import argparse

import torch
import einops

from peft import LoraConfig
from pixcell_transformer_2d_lora import PixCellTransformer2DModelLoRA

from diffusers import AutoencoderKL
from diffusers import DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler

import timm
from timm import layers
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser(description="Train LoRA on IHC-stained images (optimized)")
    parser.add_argument("--dataset", type=str, choices=["MIST", "HER2Match", "CUSTOM_NPZ"], required=True)
    parser.add_argument("--root_dir", type=str, default="/path/to/data/")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--stain", type=str, default="")

    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)

    parser.add_argument("--mixed_precision", type=str, default=None, choices=[None, "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)

    parser.add_argument("--output_dir", type=str, default="./training_stuff")
    parser.add_argument("--uncond_prob", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="./")

    # DataLoader knobs
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true", default=True)

    # Perf knobs
    parser.add_argument("--use_tf32", action="store_true", default=False)
    parser.add_argument("--channels_last", action="store_true", default=True)
    
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path al file .pth per riprendere il training")

    args = parser.parse_args()

    if args.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Initialize accelerator early (per device)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=None,
        project_dir=os.path.join(args.output_dir, "logs"),
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    # ---- Load UNI (eval only) ----
    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    uni_model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    uni_transform = create_transform(**resolve_data_config(uni_model.pretrained_cfg, model=uni_model))
    uni_model.eval()

    # ---- Load VAE + scheduler ----
    sd3_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-large", subfolder="vae")
    scheduler = DPMSolverMultistepScheduler.from_pretrained("StonyBrook-CVLab/PixCell-1024", subfolder="scheduler")

    # ---- Create transformer + load base weights ----
    config = {
        "_class_name": "PixCellTransformer2DModel",
        "_diffusers_version": "0.32.2",
        "_name_or_path": "pixart_1024/transformer",
        "activation_fn": "gelu-approximate",
        "attention_bias": True,
        "attention_head_dim": 72,
        "attention_type": "default",
        "caption_channels": 1536,
        "caption_num_tokens": 64, 
        "cross_attention_dim": 1152,
        "dropout": 0.0,
        "in_channels": 16,
        "interpolation_scale": 2,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-06,
        "norm_num_groups": 32,
        "norm_type": "ada_norm_single",
        "num_attention_heads": 16,
        "num_embeds_ada_norm": 1000,
        "num_layers": 28,
        "out_channels": 32,
        "patch_size": 2,
        "sample_size": 128,
        "upcast_attention": False,
        "use_additional_conditions": False,
    }
    lora_transformer = PixCellTransformer2DModelLoRA(**config)
    
    ckpt_path = hf_hub_download(
        repo_id="StonyBrook-CVLab/PixCell-1024",
        filename="transformer/diffusion_pytorch_model.safetensors",
        local_dir="downloads/",
    )
    '''
    lora_transformer.load_state_dict(load_file(ckpt_path), strict=False)
    '''
    state_dict = load_file(ckpt_path)
    
    key_to_remove = "caption_projection.uncond_embedding"
    if key_to_remove in state_dict:
        print(f"[INFO] Rimuovo '{key_to_remove}' dal checkpoint per mismatch di shape (16 -> 64 token)")
        del state_dict[key_to_remove]
    
    lora_transformer.load_state_dict(state_dict, strict=False)
    # ---- Add LoRA to cross-attention layers ----
    target_modules = [
        "attn2.add_k_proj",
        "attn2.add_q_proj",
        "attn2.add_v_proj",
        "attn2.to_add_out",
        "attn2.to_k",
        "attn2.to_out.0",
        "attn2.to_q",
        "attn2.to_v",
    ]
    rank = 4
    transformer_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.0,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    lora_transformer.add_adapter(transformer_lora_config)
    
    start_epoch = 0
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"\n[INFO] checkpoint: {args.resume_from_checkpoint}\n")
        state_dict_to_resume = torch.load(args.resume_from_checkpoint, map_location="cpu")
        lora_transformer.load_state_dict(state_dict_to_resume, strict=False)
        
        try:
            nome_file = os.path.basename(args.resume_from_checkpoint)
            start_epoch = int(nome_file.split("_lora_")[-1].split(".")[0])
        except Exception as e:
    # ==========================================

    if args.gradient_checkpointing and hasattr(lora_transformer, "enable_gradient_checkpointing"):
        lora_transformer.enable_gradient_checkpointing()

    # ---- DataLoader ----
    if args.dataset == "MIST":
        from mist_dataset import MISTDataset
        dataset = MISTDataset(root_dir=args.root_dir, split=args.split, stain=args.stain)
    elif args.dataset == "HER2Match":
        from her2match_dataset import HER2MatchDataset
        dataset = HER2MatchDataset(root_dir=args.root_dir, split=args.split)
    elif args.dataset == "CUSTOM_NPZ":
        from oldNpz_dataset import NPZFolderDataset
        ihc_folder = os.path.expanduser("/home/sg510849/giuSpathis/data/ihc_filtered")
        dataset = NPZFolderDataset(folder_path=ihc_folder, key="arr_0", max_per_file=300)

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0 and args.persistent_workers),
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # ---- Freeze VAE; keep only LoRA trainable ----
    sd3_vae.requires_grad_(False)
    uni_model.requires_grad_(False)

    # Trainable params
    lora_parameters = [p for p in lora_transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_parameters, lr=args.learning_rate)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=None,
        num_cycles=1,
        power=0,
    )

    # Prepare with accelerator
    lora_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_transformer, optimizer, train_dataloader, lr_scheduler
    )
    vae = accelerator.prepare_model(sd3_vae, evaluation_mode=True)
    uni_model = accelerator.prepare_model(uni_model, evaluation_mode=True)

    # ---- Precompute constants on device ----
    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0.0)

    # alphas_cumprod su GPU (evita copie ad ogni step)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    # optional channels_last
    if args.channels_last:
        lora_transformer.to(memory_format=torch.channels_last)

    global_step = start_epoch * len(train_dataloader)

    for _ in range(global_step):
        lr_scheduler.step()

    for epoch in range(start_epoch, args.num_epochs):
        lora_transformer.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(lora_transformer):
                he, ihc = batch  # he unused in unpaired
                ihc = ihc.to(device, non_blocking=True)

                if args.channels_last:
                    ihc = ihc.contiguous(memory_format=torch.channels_last)

                bs = ihc.shape[0]

                with accelerator.autocast():
                    # ---- UNI embeddings ----
                    # 1024x1024 -> 16 patches 256x256
                    '''
                    uni_patches = einops.rearrange(
                        ihc, "b c (d1 h) (d2 w) -> (b d1 d2) c h w", d1=4, d2=4
                    )
                    '''
                    uni_patches = einops.rearrange(
                        ihc, "b c (d1 h) (d2 w) -> (b d1 d2) c h w", d1=8, d2=8
                    )
                    uni_input = uni_transform(uni_patches)
                    uni_input = uni_input.to(device, non_blocking=True)

                    with torch.inference_mode():
                        uni_emb_ihc = uni_model(uni_input)
                    uni_emb_ihc = uni_emb_ihc.unsqueeze(0).reshape(bs, 64, -1) 

                    # ---- Encode IHC -> latents ----
                    ihc_in = (2.0 * (ihc - 0.5)).to(dtype=vae.dtype)
                    ihc_latents = vae.encode(ihc_in).latent_dist.sample()
                    ihc_latents = (ihc_latents - vae_shift) * vae_scale

                    # ---- Noise ----
                    t = torch.randint(0, 1000, (bs,), device=device, dtype=torch.int64)
                    atbar = alphas_cumprod[t].view(bs, 1, 1, 1)
                    epsilon = torch.randn_like(ihc_latents)
                    noisy_latents = torch.sqrt(atbar) * ihc_latents + torch.sqrt(1.0 - atbar) * epsilon

                    # ---- Classifier-free drop ----
                    if args.uncond_prob > 0:
                        uncond = lora_transformer.caption_projection.uncond_embedding.clone().tile(bs, 1, 1)
                        mask = (torch.rand((bs, 1, 1), device=device) < args.uncond_prob).float()
                        uni_emb_ihc = (1.0 - mask) * uni_emb_ihc + mask * uncond

                    # ---- Denoiser ----
                    epsilon_pred = lora_transformer(
                        noisy_latents,
                        encoder_hidden_states=uni_emb_ihc,
                        timestep=t,
                        return_dict=False,
                    )[0]

                    loss = ((epsilon_pred[:, :16, :, :] - epsilon) ** 2).mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_parameters, 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": float(loss.detach().item()), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        # ---- Save ----
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.makedirs(args.save_dir, exist_ok=True)
            lora_transformer_unwrapped = accelerator.unwrap_model(lora_transformer)
            save_name = f"{args.dataset}_{args.stain}_lora_{epoch+1}.pth"
            torch.save(lora_transformer_unwrapped.state_dict(), os.path.join(args.save_dir, save_name))

    accelerator.end_training()


if __name__ == "__main__":
    main()


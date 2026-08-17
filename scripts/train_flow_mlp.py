#!/usr/bin/env python3

import os
import numpy as np
import torch
import einops

from resmlp import SimpleMLP
from torch.utils.data import DataLoader
from tqdm import tqdm

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from huggingface_hub import hf_hub_download

import argparse


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train rectified flow MLP between H&E and IHC UNI embeddings"
    )

    # --- dataset ---
    parser.add_argument(
        "--dataset", type=str,
        choices=["MIST", "HER2Match", "CUSTOM_NPZ_PAIRED"],
        required=True,
    )
    # MIST / HER2Match
    parser.add_argument("--root_dir",        type=str, default="/path/to/data/")
    parser.add_argument("--split",           type=str, default="train")
    parser.add_argument("--stain",           type=str, default="CK7")

    # CUSTOM_NPZ_PAIRED
    parser.add_argument("--he_dir",          type=str, default=None,
                        help="Cartella he_*.npz  (richiesto per CUSTOM_NPZ_PAIRED)")
    parser.add_argument("--ihc_paired_dir",  type=str, default=None,
                        help="Cartella ihc_*.npz paired (richiesto per CUSTOM_NPZ_PAIRED)")
    parser.add_argument("--max_per_file",    type=int, default=300)

    # --- training ---
    parser.add_argument("--device",          type=str, default="cuda")
    parser.add_argument("--train_batch_size",type=int, default=4)
    parser.add_argument("--num_epochs",      type=int, default=100)
    parser.add_argument("--learning_rate",   type=float, default=1e-4)
    parser.add_argument("--save_every",      type=int, default=25)
    parser.add_argument("--save_dir",        type=str, default="./checkpoints_mlp")
    parser.add_argument("--num_workers",     type=int, default=4)

    # --- warm start ---
    parser.add_argument(
        "--warmstart_ckpt", type=str, default=None,
        help="Path a un checkpoint MLP .pth da cui iniziare "
             "(es. downloads/mist_her2_mlp.pth). "
             "Se None usa pesi random."
    )
    parser.add_argument(
        "--warmstart_target", type=str, default="mist_her2",
        choices=["mist_her2", "mist_er", "mist_pr", "mist_ki67", "her2match"],
        help="Se --warmstart_ckpt non è specificato, scarica questo checkpoint da HuggingFace."
    )
    parser.add_argument(
        "--no_warmstart", action="store_true",
        help="Forza training da zero anche se --warmstart_ckpt è specificato."
    )

    # --- perf ---
    parser.add_argument("--use_bf16",  action="store_true")
    parser.add_argument("--use_tf32",  action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.use_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32        = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    autocast_dtype = (
        torch.bfloat16 if (device.type == "cuda" and args.use_bf16)
        else torch.float16
    )
    use_autocast = device.type == "cuda"

    os.makedirs(args.save_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1.  UNI
    # ------------------------------------------------------------------ #
    timm_kwargs = {
        "img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24,
        "init_values": 1e-5, "embed_dim": 1536, "mlp_ratio": 2.66667 * 2,
        "num_classes": 0, "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8, "dynamic_img_size": True,
    }
    uni_model     = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    uni_transform = create_transform(**resolve_data_config(uni_model.pretrained_cfg, model=uni_model))
    uni_model.eval().to(device)

    # ------------------------------------------------------------------ #
    # 2.  Dataset / DataLoader
    # ------------------------------------------------------------------ #
    print(f"[INFO] Dataset: {args.dataset}")

    if args.dataset == "MIST":
        from mist_dataset import MISTDataset
        dataset = MISTDataset(root_dir=args.root_dir, split=args.split, stain=args.stain)

    elif args.dataset == "HER2Match":
        from her2match_dataset import HER2MatchDataset
        dataset = HER2MatchDataset(root_dir=args.root_dir, split=args.split)

    elif args.dataset == "CUSTOM_NPZ_PAIRED":
        if args.he_dir is None or args.ihc_paired_dir is None:
            raise ValueError(
                "Per CUSTOM_NPZ_PAIRED specificare --he_dir e --ihc_paired_dir"
            )
        from npz_dataset import NPZFolderDataset
        dataset = NPZFolderDataset(
            folder_path      = args.he_dir,
            mask_folder_path = args.ihc_paired_dir,
            key              = "arr_0",
            max_per_file     = args.max_per_file,
            verbose          = True,
        )

    train_dataloader = DataLoader(
        dataset,
        batch_size  = args.train_batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = (device.type == "cuda"),
        persistent_workers = False,
        prefetch_factor    = 2 if args.num_workers > 0 else None,
    )

    # ------------------------------------------------------------------ #
    # 3.  MLP
    # ------------------------------------------------------------------ #
    uni_mlp = SimpleMLP(
        in_channels       = 1536,
        time_embed_dim    = 1024,
        model_channels    = 1024,
        bottleneck_channels = 1024,
        out_channels      = 1536,
        num_res_blocks    = 6,
    ).to(device)

    print(f"[INFO] MLP params: {sum(p.numel() for p in uni_mlp.parameters()):,d}")

    # --- warm start ---
    if not args.no_warmstart:
        ckpt_path = args.warmstart_ckpt

        if ckpt_path is None:
            # scarica da HuggingFace
            mlp_map = {
                "mist_her2":  "ckpts/mlp/mist_her2_mlp.pth",
                "mist_er":    "ckpts/mlp/mist_er_mlp.pth",
                "mist_pr":    "ckpts/mlp/mist_pr_mlp.pth",
                "mist_ki67":  "ckpts/mlp/mist_ki67_mlp.pth",
                "her2match":  "ckpts/mlp/her2match_mlp.pth",
            }
            print(f"[INFO] warm-start checkpoint: {args.warmstart_target} …")
            ckpt_path = hf_hub_download(
                repo_id    = "StonyBrook-CVLab/pixcell-virtual-staining",
                filename   = mlp_map[args.warmstart_target],
                local_dir  = "downloads/",
            )

        print(f"[INFO] Warm-start da: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = uni_mlp.load_state_dict(state, strict=False)
        if missing:
            print(f"  [WARN] Chiavi mancanti  : {missing}")
        if unexpected:
            print(f"  [WARN] Chiavi inattese  : {unexpected}")
    else:
        print("[INFO] Training from scratch (no warm-start).")

    # ------------------------------------------------------------------ #
    # 4.  Optimizer
    # ------------------------------------------------------------------ #
    opt = torch.optim.AdamW(uni_mlp.parameters(), lr=args.learning_rate)

    # ------------------------------------------------------------------ #
    # 5.  Training loop
    # ------------------------------------------------------------------ #
    uni_mlp.train()
    losses = []

    for epoch in range(args.num_epochs):
        print(f"\nEpoch [{epoch+1}/{args.num_epochs}]")
        bar = tqdm(train_dataloader)

        for batch in bar:
            he, ihc = batch
            bs = he.shape[0]

            # ---- UNI embeddings ----
            # 1024x1024 -> 16 patch 256x256 
            uni_patches_he = einops.rearrange(
                he, "b c (d1 h) (d2 w) -> (b d1 d2) c h w", d1=4, d2=4
            )
            uni_patches_ihc = einops.rearrange(
                ihc, "b c (d1 h) (d2 w) -> (b d1 d2) c h w", d1=4, d2=4
            )

            uni_input_he  = uni_transform(uni_patches_he)
            uni_input_ihc = uni_transform(uni_patches_ihc)

            with torch.inference_mode():
                with torch.autocast(device_type=device.type,
                                    dtype=autocast_dtype,
                                    enabled=use_autocast):
                    uni_emb = uni_model(
                        torch.cat((uni_input_he, uni_input_ihc), dim=0).to(device)
                    )
            uni_emb_he, uni_emb_ihc = torch.chunk(uni_emb.float(), chunks=2, dim=0)

            # reshape: (b*16, 1536)
            uni_emb_he  = uni_emb_he .reshape(bs * 16, 1536)
            uni_emb_ihc = uni_emb_ihc.reshape(bs * 16, 1536)

            # ---- Flow matching ----
            # xt = t*ihc + (1-t)*he   target = ihc - he
            batch_size = uni_emb_he.shape[0]
            t      = torch.rand((batch_size,), device=device).view(-1, 1)
            xt     = t * uni_emb_ihc + (1 - t) * uni_emb_he
            target = uni_emb_ihc - uni_emb_he

            with torch.autocast(device_type=device.type,
                                dtype=autocast_dtype,
                                enabled=use_autocast):
                pred = uni_mlp(xt, 999.0 * t.view(-1))

            loss = ((pred.float() - target) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(uni_mlp.parameters(), 1.0)
            opt.step()

            losses.append(loss.item())
            if len(losses) > 100:
                losses = losses[-100:]
            bar.set_postfix({"loss": f"{np.mean(losses):.5f}"})

        if (epoch + 1) % args.save_every == 0:
            name = f"CUSTOM_NPZ_PAIRED_{args.stain}_mlp_{epoch+1}.pth"
            path = os.path.join(args.save_dir, name)
            torch.save(uni_mlp.state_dict(), path)
            print(f"  [SAVE] {path}")

    final_name = f"CUSTOM_NPZ_PAIRED_{args.stain}_mlp_final.pth"
    final_path = os.path.join(args.save_dir, final_name)
    torch.save(uni_mlp.state_dict(), final_path)
    print(f"\n[DONE] Checkpoint finale: {final_path}")


if __name__ == "__main__":
    main()

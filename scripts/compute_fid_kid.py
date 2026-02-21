#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

try:
    import pandas as pd
except ImportError:
    pd = None

NPZ_EXTS = {".npz"}


def _tile_to_uint8_numpy(tile) -> np.ndarray:
    if not isinstance(tile, np.ndarray):
        tile = np.array(tile)
    if tile.dtype == object:
        tile = np.array(tile.tolist(), dtype=np.uint8)
    elif tile.dtype != np.uint8:
        tile = tile.astype(np.uint8)
    return tile


def _ensure_hwc_rgb_uint8(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    elif x.ndim == 3:
        if x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
            x = np.transpose(x, (1, 2, 0))
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
    else:
        raise ValueError(f"Unexpected image shape {x.shape}")

    if x.dtype != np.uint8:
        x = x.astype(np.float32)
        mx = float(np.nanmax(x)) if x.size else 0.0
        if mx <= 1.5:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)

    if x.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels, got {x.shape}")
    return x



def list_npz_files(folder: str) -> List[str]:
    p = Path(os.path.expanduser(folder))
    if not p.exists():
        raise FileNotFoundError(str(p))
    files = sorted([str(x) for x in p.iterdir() if x.is_file() and x.suffix.lower() in NPZ_EXTS])
    if not files:
        raise RuntimeError(f"No .npz found in {p}")
    return files


def list_npz_basenames(folder: str) -> List[str]:
    return [Path(f).name for f in list_npz_files(folder)]


def resolve_files_by_basenames(folder: str, basenames: List[str]) -> List[str]:
    p = Path(os.path.expanduser(folder))
    out = []
    missing = []
    for bn in basenames:
        fp = p / bn
        if fp.exists():
            out.append(str(fp))
        else:
            missing.append(bn)
    
    return out


def choose_common_fake_basenames(fake_folders_abs: List[str], n_files: int) -> List[str]:
  
    sets = []
    for f in fake_folders_abs:
        bns = set(list_npz_basenames(f))
        sets.append(bns)

    common = set.intersection(*sets) if sets else set()
    common_sorted = sorted(list(common))

    
    return common_sorted[:n_files]


def per_file_counts(target_total: int, n_files: int) -> List[int]:
    base = target_total // n_files
    rem = target_total % n_files
    return [base + (1 if i < rem else 0) for i in range(n_files)]


def load_npz_tiles_pixcell_range(npz_path: str, key: str, start: int, count: int, mmap: bool) -> List[np.ndarray]:
    data = np.load(npz_path, mmap_mode="r" if mmap else None, allow_pickle=True)
    if key not in data:
        raise KeyError(f"Key '{key}' not found in {npz_path}. Keys={list(data.keys())}")
    arr = data[key]
    n = int(arr.shape[0])

    s = max(0, int(start))
    if s >= n:
        return []

    t = min(int(count), n - s)
    out: List[np.ndarray] = []
    for i in range(s, s + t):
        tile = arr[i, 0]
        tile = _tile_to_uint8_numpy(tile)
        tile = _ensure_hwc_rgb_uint8(tile)
        out.append(tile)
    return out


def gather_npz_from_file_list(
    file_list: List[str],
    npz_key: str,
    target_total: int,
    mmap: bool = True,
    verbose: bool = True,
) -> List[np.ndarray]:
    
    n_files = len(file_list)
    counts = per_file_counts(int(target_total), n_files)
    if verbose:
        print(f"[INFO] target_total={target_total}, npz_files={n_files}, per_file={counts}")

    out: List[np.ndarray] = []
    taken_firstpass = [0] * n_files

    # PASS 1: split uniforme
    for idx, (f, k) in enumerate(zip(file_list, counts)):
        tiles = load_npz_tiles_pixcell_range(f, npz_key, start=0, count=k, mmap=mmap)
        out.extend(tiles)
        taken_firstpass[idx] = len(tiles)

    deficit = target_total - len(out)
    if deficit > 0 and verbose:
        print(f"[INFO] Short by {deficit} images (some npz smaller). Filling within the same selected files...")

    if deficit > 0:
        for idx, f in enumerate(file_list):
            if deficit <= 0:
                break
            extra = load_npz_tiles_pixcell_range(f, npz_key, start=taken_firstpass[idx], count=deficit, mmap=mmap)
            out.extend(extra)
            deficit = target_total - len(out)

    if verbose:
        print(f"[INFO] After fill: {len(out)} images")

    

    if len(out) > target_total:
        out = out[:target_total]
    return out


# -----------------------------
# Crop sampling (seeded)
# -----------------------------
def random_crops(img: np.ndarray, crop_size: int, num_crops: int, rng: np.random.Generator) -> List[np.ndarray]:
    h, w, _ = img.shape
    if h < crop_size or w < crop_size:
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        img = np.pad(
            img,
            ((pad_h // 2, pad_h - pad_h // 2),
             (pad_w // 2, pad_w - pad_w // 2),
             (0, 0)),
            mode="reflect"
        )
        h, w, _ = img.shape

    out = []
    for _ in range(num_crops):
        y = int(rng.integers(0, h - crop_size + 1))
        x = int(rng.integers(0, w - crop_size + 1))
        out.append(img[y:y + crop_size, x:x + crop_size, :])
    return out


# -----------------------------
# Vanilla FID on full tiles (Inception-v3 features)
# -----------------------------
def compute_full_fid(
    real_imgs: List[np.ndarray],
    fake_imgs: List[np.ndarray],
    device: str,
) -> float:
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    def update(imgs: List[np.ndarray], real: bool):
        batch = []
        for im in imgs:
            t = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0)  # uint8 CHW
            batch.append(t)
            if len(batch) >= 32:
                x = torch.cat(batch, dim=0).to(device)
                fid.update(x, real=real)
                batch = []
        if batch:
            x = torch.cat(batch, dim=0).to(device)
            fid.update(x, real=real)

    update(real_imgs, real=True)
    update(fake_imgs, real=False)
    return float(fid.compute().detach().cpu().item())


# -----------------------------
# Crop FID + KID (Inception-based, seeded)
# -----------------------------
def make_kid_metric(device: str, subset_size: int, subsets: int, normalize: bool = True):
    try:
        return KernelInceptionDistance(subset_size=subset_size, subsets=subsets, normalize=normalize).to(device)
    except Exception:
        pass
    try:
        return KernelInceptionDistance(subset_size=subset_size, num_subsets=subsets, normalize=normalize).to(device)
    except Exception:
        pass
    return KernelInceptionDistance(subset_size=subset_size, normalize=normalize).to(device)


def compute_crop_fid_kid(
    real_imgs: List[np.ndarray],
    fake_imgs: List[np.ndarray],
    crop_size: int,
    crops_per_image: int,
    seed: int,
    device: str,
    kid_subsets: int,
    kid_subset_size: int,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    kid = make_kid_metric(device=device, subset_size=kid_subset_size, subsets=kid_subsets, normalize=True)

    def update_metrics(imgs: List[np.ndarray], real: bool):
        batch = []
        for im in imgs:
            for cr in random_crops(im, crop_size, crops_per_image, rng):
                t = torch.from_numpy(cr).permute(2, 0, 1).unsqueeze(0)  # uint8 CHW
                batch.append(t)
                if len(batch) >= 32:
                    x = torch.cat(batch, dim=0).to(device)
                    fid.update(x, real=real)
                    kid.update(x, real=real)
                    batch = []
        if batch:
            x = torch.cat(batch, dim=0).to(device)
            fid.update(x, real=real)
            kid.update(x, real=real)

    update_metrics(real_imgs, real=True)
    update_metrics(fake_imgs, real=False)

    cropfid = float(fid.compute().detach().cpu().item())
    kid_out = kid.compute()

    if isinstance(kid_out, (tuple, list)) and len(kid_out) == 2:
        kid_mean, kid_std = kid_out
    elif isinstance(kid_out, dict) and "kid_mean" in kid_out and "kid_std" in kid_out:
        kid_mean, kid_std = kid_out["kid_mean"], kid_out["kid_std"]
    else:
        kid_mean, kid_std = kid_out, torch.tensor(0.0)

    kid_mean = float(kid_mean.detach().cpu().item())
    kid_std = float(kid_std.detach().cpu().item())
    return cropfid, kid_mean, kid_std


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", type=str, required=True)
    ap.add_argument("--real_root", type=str, required=True)
    ap.add_argument("--folders", type=str,
                    default="mist_er,mist_pr,CUSTOM_NPZ_CK7_lora_20,CUSTOM_NPZ_CK7_lora_20_noFlow, mist_ki67")
    ap.add_argument("--npz_key", type=str, default="arr_0")

    ap.add_argument("--target_total", type=int, default=3000)
    ap.add_argument("--fake_n_files", type=int, default=7)

    ap.add_argument("--no_mmap", action="store_true")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--crops_per_image", type=int, default=4)

    ap.add_argument("--kid_subsets", type=int, default=50)
    ap.add_argument("--kid_subset_size", type=int, default=100)

    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--out_xlsx", type=str, default="results_metrics_single_sheet.xlsx")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    mmap = not args.no_mmap
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    
    folders = [x.strip() for x in args.folders.split(",") if x.strip()]
    
    fake_folders_abs = [str(Path(os.path.expanduser(args.gen_root)) / f) for f in folders]

    common_basenames = choose_common_fake_basenames(fake_folders_abs, args.fake_n_files)
    print(f"[INFO] FAKE common basenames ({len(common_basenames)}): {common_basenames}")

    real_files_all = list_npz_files(args.real_root)
    real_imgs = gather_npz_from_file_list(
        real_files_all,
        npz_key=args.npz_key,
        target_total=args.target_total,
        mmap=mmap,
        verbose=True,
    )
    print(f"[INFO] REAL: {len(real_imgs)}")

    fake_cache: Dict[str, List[np.ndarray]] = {}

    for name in folders:
        fake_path = str(Path(os.path.expanduser(args.gen_root)) / name)
        file_list = resolve_files_by_basenames(fake_path, common_basenames)

        fake_imgs = gather_npz_from_file_list(
            file_list,
            npz_key=args.npz_key,
            target_total=args.target_total,
            mmap=mmap,
            verbose=True,
        )
        fake_cache[name] = fake_imgs

    rows: List[Dict[str, Any]] = []

    for name in folders:
        fake_path = str(Path(os.path.expanduser(args.gen_root)) / name)

        # Vanilla FID on full tiles 
        fid_full = compute_full_fid(
            real_imgs=real_imgs,
            fake_imgs=fake_cache[name],
            device=device,
        )

        for sd in seeds:
            print(f"\n[RUN] model={name} seed={sd}")

            cropfid, kid_mean, kid_std = compute_crop_fid_kid(
                real_imgs=real_imgs,
                fake_imgs=fake_cache[name],
                crop_size=args.crop_size,
                crops_per_image=args.crops_per_image,
                seed=sd,
                device=device,
                kid_subsets=args.kid_subsets,
                kid_subset_size=args.kid_subset_size,
            )

            rows.append({
                "model_folder": name,
                "real_path": args.real_root,
                "fake_path": fake_path,
                "target_total": args.target_total,
                "n_real": len(real_imgs),
                "n_fake": len(fake_cache[name]),
                "npz_key": args.npz_key,
                "real_npz_files_used": len(real_files_all),
                "fake_npz_files_used": args.fake_n_files,
                "fake_basenames": ";".join(common_basenames),
                "seed": sd,
                "crop_size": args.crop_size,
                "crops_per_image": args.crops_per_image,
                "kid_subsets": args.kid_subsets,
                "kid_subset_size": args.kid_subset_size,
                "FID": fid_full,
                "CropFID": cropfid,
                "KID_mean": kid_mean,
                "KID_std": kid_std,
                "row_type": "seed",
            })

    
    df = pd.DataFrame(rows)

    # Summary per model: mean/std across seeds
    metric_cols = ["FID", "CropFID", "KID_mean", "KID_std"]
    agg = df.groupby("model_folder")[metric_cols].agg(["mean", "std"]).reset_index()

    summary_rows = []
    for _, r in agg.iterrows():
        model = r[("model_folder", "")]
        mean_vals = {m: float(r[(m, "mean")]) for m in metric_cols}
        std_vals = {m: float(r[(m, "std")]) for m in metric_cols}

        base_info = {
            "model_folder": model,
            "real_path": args.real_root,
            "fake_path": str(Path(os.path.expanduser(args.gen_root)) / model),
            "target_total": args.target_total,
            "n_real": len(real_imgs),
            "n_fake": len(fake_cache[model]),
            "npz_key": args.npz_key,
            "real_npz_files_used": len(real_files_all),
            "fake_npz_files_used": args.fake_n_files,
            "fake_basenames": ";".join(common_basenames),
            "crop_size": args.crop_size,
            "crops_per_image": args.crops_per_image,
            "kid_subsets": args.kid_subsets,
            "kid_subset_size": args.kid_subset_size,
        }

        mean_row = dict(base_info)
        mean_row.update({"seed": "MEAN", "row_type": "summary_mean", **mean_vals})
        std_row = dict(base_info)
        std_row.update({"seed": "STD", "row_type": "summary_std", **std_vals})
        summary_rows.extend([mean_row, std_row])

    df_summary = pd.DataFrame(summary_rows)
    df_out = pd.concat([df, df_summary], ignore_index=True)

    # sort: seed rows first, then mean, then std
    def _ord(rt: str) -> int:
        if rt == "seed":
            return 0
        if rt == "summary_mean":
            return 1
        return 2

    df_out["__ord"] = df_out["row_type"].map(_ord)

    def seed_to_num(s):
        try:
            return int(s)
        except Exception:
            return 10**9

    df_out["__seednum"] = df_out["seed"].apply(seed_to_num)
    df_out = df_out.sort_values(["__ord", "model_folder", "__seednum"]).drop(columns=["__ord", "__seednum"])

    df_out.to_excel(args.out_xlsx, index=False)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3

import os
import re
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -----------------------------
# NPZ I/O
# -----------------------------
def pick_array(npz):
    if "arr_0" in npz.files:
        return npz["arr_0"], "arr_0"
    k = npz.files[0]
    return npz[k], k

def load_npz_array(npz_path):
    with np.load(npz_path, allow_pickle=True) as npz:
        arr, key = pick_array(npz)
    
    return arr

def to_uint8_rgb(img, is_bgr=False):
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[..., :3]
    if is_bgr and arr.shape[-1] == 3:
        arr = arr[..., ::-1]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)
    return arr

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# -----------------------------
# Filename matching 
# he_XXXX.npz <-> ihc_XXXX.npz
# -----------------------------
_HE_RE = re.compile(r"^(he_.+)\.npz$")
_IHC_FROM_HE = lambda he_stem: he_stem.replace("he_", "ihc_", 1)

def list_he_files(he_dir):
    return sorted([fn for fn in os.listdir(he_dir) if fn.endswith(".npz") and fn.startswith("he_")])

def build_ihc_path(ckpt_dir, he_fn):
    he_stem = os.path.splitext(he_fn)[0]
    ihc_stem = _IHC_FROM_HE(he_stem)
    return os.path.join(ckpt_dir, ihc_stem + ".npz")

# -----------------------------
# Tissue mask (HSV) + central crop
# -----------------------------
def tissue_fraction_rgb_uint8(rgb):
    im = Image.fromarray(rgb, mode="RGB").convert("HSV")
    hsv = np.asarray(im).astype(np.float32)
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    tissue = (v < 0.92) & (s > 0.05)
    return float(tissue.mean())

def center_crop(rgb, size):
    h, w = rgb.shape[:2]
    if size <= 0 or size > min(h, w):
        return rgb
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return rgb[y0:y0+size, x0:x0+size]

# -----------------------------
# Proxies
# -----------------------------
def hsv_saturation_std(rgb):
    im = Image.fromarray(rgb, mode="RGB").convert("HSV")
    hsv = np.asarray(im).astype(np.float32)
    s = hsv[..., 1] / 255.0
    return float(np.std(s))

def laplacian_var(rgb):
    x = rgb.astype(np.float32)
    y = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    k = np.array([[0,  1, 0],
                  [1, -4, 1],
                  [0,  1, 0]], dtype=np.float32)
    yp = np.pad(y, 1, mode="edge")
    out = (
        k[0,0]*yp[:-2,:-2] + k[0,1]*yp[:-2,1:-1] + k[0,2]*yp[:-2,2:] +
        k[1,0]*yp[1:-1,:-2] + k[1,1]*yp[1:-1,1:-1] + k[1,2]*yp[1:-1,2:] +
        k[2,0]*yp[2:,:-2] + k[2,1]*yp[2:,1:-1] + k[2,2]*yp[2:,2:]
    )
    return float(np.var(out))

def autocorr_shift_score(rgb, shift=16):
    x = rgb.astype(np.float32)
    y = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    y = (y - y.mean()) / (y.std() + 1e-6)
    if y.shape[0] <= shift or y.shape[1] <= shift:
        return 0.0
    a = y[:, :-shift]
    b = y[:, shift:]
    c = y[:-shift, :]
    d = y[shift:, :]
    s1 = float(np.mean(a * b))
    s2 = float(np.mean(c * d))
    return (s1 + s2) / 2.0

# -----------------------------
# Plot
# -----------------------------
def plot_grid(rows, col_titles, out_path, dpi=200, suptitle=None, caption=None):
    R = len(rows)
    C = len(col_titles)
    fig_h = max(2.6, 2.2 * R)
    fig_w = max(7.0, 2.2 * C)
    fig, axes = plt.subplots(R, C, figsize=(fig_w, fig_h), squeeze=False)

    for r in range(R):
        for c in range(C):
            ax = axes[r, c]
            ax.imshow(rows[r][c])
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    if caption:
        fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--he_dir", required=True, help="Dir HE paired (es: data/testing/he)")
    ap.add_argument("--checkpoints", required=True, nargs="+", help="Dir IHC generated (ihc_*.npz)")
    ap.add_argument("--ckpt_names", default=None, nargs="*")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n_random", type=int, default=6)
    ap.add_argument("--tissue_min", type=float, default=0.60)
    ap.add_argument("--center_tissue_min", type=float, default=0.60)
    ap.add_argument("--center_crop", type=int, default=512)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--bgr", action="store_true")
    ap.add_argument("--rank_checkpoint", default=None)
    ap.add_argument("--scan_per_file", type=int, default=120)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    rng = np.random.default_rng(args.seed)

    ckpt_dirs = args.checkpoints
    if args.ckpt_names is None or len(args.ckpt_names) == 0:
        ckpt_names = [os.path.basename(d.rstrip("/")) for d in ckpt_dirs]
    else:
        ckpt_names = args.ckpt_names

    he_files = list_he_files(args.he_dir)
    
    valid_he = []
    for he_fn in he_files:
        ok = True
        for d in ckpt_dirs:
            if not os.path.exists(build_ihc_path(d, he_fn)):
                ok = False
                break
        if ok:
            valid_he.append(he_fn)

    
    def passes_tissue_filters(rgb):
        tf_full = tissue_fraction_rgb_uint8(rgb)
        if tf_full < args.tissue_min:
            return False
        if args.center_tissue_min > 0:
            cc = center_crop(rgb, args.center_crop)
            tf_c = tissue_fraction_rgb_uint8(cc)
            if tf_c < args.center_tissue_min:
                return False
        return True

    # ------------------ FIG 2: random ------------------
    chosen = rng.choice(valid_he, size=args.n_random, replace=False)
    fig2_rows = []

    for he_fn in chosen:
        he_path = os.path.join(args.he_dir, he_fn)
        he_arr = load_npz_array(he_path)

        ihc_arrs = []
        minN = he_arr.shape[0]
        for d in ckpt_dirs:
            a = load_npz_array(build_ihc_path(d, he_fn))
            ihc_arrs.append(a)
            minN = min(minN, a.shape[0])

        if minN == 0:
            continue

        idxs = rng.permutation(minN)
        chosen_i = None
        he_img = None
        for i in idxs[:min(minN, 400)]:
            tmp = to_uint8_rgb(he_arr[i, 0], is_bgr=args.bgr)
            if passes_tissue_filters(tmp):
                chosen_i = int(i)
                he_img = tmp
                break
        if chosen_i is None:
            chosen_i = int(idxs[0])
            he_img = to_uint8_rgb(he_arr[chosen_i, 0], is_bgr=args.bgr)

        row = [he_img]
        for a in ihc_arrs:
            row.append(to_uint8_rgb(a[chosen_i, 0], is_bgr=args.bgr))
        fig2_rows.append(row)

    col_titles = ["H&E"] + [f"IHC gen: {n}" for n in ckpt_names]

    fig2_caption = (f"Randomly sampled (seed={args.seed}), one patch per WSI/file, "
                    f"tissue >= {int(args.tissue_min*100)}% "
                    f"(center >= {int(args.center_tissue_min*100)}% on {args.center_crop}x{args.center_crop}), "
                    f"same crop across checkpoints (same tile index).")
    fig2_path = os.path.join(args.out_dir, "Figure2_random6.png")
    plot_grid(fig2_rows, col_titles, fig2_path, dpi=200,
              suptitle="Figure 2 — Random samples", caption=fig2_caption)

    # ------------------ FIG 3: best/worst ------------------
    if args.rank_checkpoint is None:
        rank_idx = 0
    else:
        if args.rank_checkpoint.isdigit():
            rank_idx = int(args.rank_checkpoint)
        else:
            rank_idx = ckpt_names.index(args.rank_checkpoint)
        
    rank_name = ckpt_names[rank_idx]
    rank_dir = ckpt_dirs[rank_idx]

    candidates = []  # (sat_std, ac, lap, he_fn, tile_i)
    for he_fn in valid_he:
        he_path = os.path.join(args.he_dir, he_fn)
        he_arr = load_npz_array(he_path)
        gen_arr = load_npz_array(build_ihc_path(rank_dir, he_fn))
        minN = min(he_arr.shape[0], gen_arr.shape[0])
        if minN == 0:
            continue

        scan_n = min(args.scan_per_file, minN)
        idxs = rng.choice(minN, size=scan_n, replace=False)

        for i in idxs:
            he_img = to_uint8_rgb(he_arr[i, 0], is_bgr=args.bgr)
            if not passes_tissue_filters(he_img):
                continue

            gen_img = to_uint8_rgb(gen_arr[i, 0], is_bgr=args.bgr)
            sat_std = hsv_saturation_std(gen_img)
            ac = autocorr_shift_score(gen_img, shift=16)
            lap = laplacian_var(gen_img)
            candidates.append((sat_std, ac, lap, he_fn, int(i)))

    

    sat = np.array([c[0] for c in candidates], dtype=np.float32)
    ac = np.array([c[1] for c in candidates], dtype=np.float32)
    lap = np.array([c[2] for c in candidates], dtype=np.float32)

    sat_z = (sat - sat.mean()) / (sat.std() + 1e-6)
    ac_z = (ac - ac.mean()) / (ac.std() + 1e-6)
    lap_z = (lap - lap.mean()) / (lap.std() + 1e-6)
    lap_extreme = np.abs(lap_z)

    badness = (-sat_z) + (ac_z) + (0.5 * lap_extreme)
    order = np.argsort(badness)
    best = order[:args.topk]
    worst = order[-args.topk:][::-1]

    def row_for(he_fn, tile_i):
        he_path = os.path.join(args.he_dir, he_fn)
        he_arr = load_npz_array(he_path)
        row = [to_uint8_rgb(he_arr[tile_i, 0], is_bgr=args.bgr)]
        for d in ckpt_dirs:
            a = load_npz_array(build_ihc_path(d, he_fn))
            row.append(to_uint8_rgb(a[tile_i, 0], is_bgr=args.bgr))
        return row

    fig3_rows = []
    for j in best:
        _, _, _, he_fn, tile_i = candidates[int(j)]
        fig3_rows.append(row_for(he_fn, tile_i))
    for j in worst:
        _, _, _, he_fn, tile_i = candidates[int(j)]
        fig3_rows.append(row_for(he_fn, tile_i))

    fig3_caption = (f"Best/Worst selected on checkpoint '{rank_name}' using proxies: "
                    f"low HSV saturation variance (global tint), high autocorrelation (repetition), "
                    f"and extreme Laplacian variance (blur/oversharp). "
                    f"Seed={args.seed}, tissue>={int(args.tissue_min*100)}%, "
                    f"center>={int(args.center_tissue_min*100)}% ({args.center_crop}x{args.center_crop}).")
    fig3_path = os.path.join(args.out_dir, f"Figure3_best_worst_rank_{rank_name}.png")
    plot_grid(fig3_rows, col_titles, fig3_path, dpi=200,
              suptitle=f"Figure 3 — Best (top {args.topk}) / Worst (top {args.topk}) on {rank_name}",
              caption=fig3_caption)

    print(" -", fig2_path)
    print(" -", fig3_path)

if __name__ == "__main__":
    main()


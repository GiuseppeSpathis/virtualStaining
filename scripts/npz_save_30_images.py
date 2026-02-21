#!/usr/bin/env python3
import os
import argparse
import numpy as np
from PIL import Image

def pick_array(npz):
    if "arr_0" in npz.files:
        return npz["arr_0"], "arr_0"
    
    k = npz.files[0]
    return npz[k], k

def to_uint8_rgb(img, is_bgr=False):
    arr = np.asarray(img)
    
    if arr.ndim == 2:  # grayscale -> RGB
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:  # RGBA -> RGB
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--prefix", default="tile")
    ap.add_argument("--ext", default="png", choices=["png", "jpg", "jpeg"])
    ap.add_argument("--force_label", default=None)
    ap.add_argument("--bgr", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with np.load(args.npz, allow_pickle=True) as npz:
        arr, key = pick_array(npz)

    
    total = arr.shape[0]
    start = max(0, args.start)
    end = min(total, start + args.n)

    

    saved = 0
    for i in range(start, end):
        img = arr[i, 0]
        label_str = ""

        if args.force_label:
             label_str = f"_{args.force_label}"
        elif arr.shape[1] >= 2:
            raw_label = arr[i, 1]
            try:
                val = float(raw_label)
                if val.is_integer():
                    label_str = f"_label{int(val)}"
                else:
                    label_str = f"_label{val:.2f}"
            except Exception:
                s_label = str(raw_label)
                if len(s_label) > 20:
                    label_str = "_complexLabel"
                else:
                    clean = "".join(x for x in s_label if x.isalnum() or x in "._-")
                    label_str = f"_label{clean}"

        rgb = to_uint8_rgb(img, is_bgr=args.bgr)
        im = Image.fromarray(rgb, mode="RGB")

        fname = f"{args.prefix}_{i:06d}{label_str}.{args.ext}"
        out_path = os.path.join(args.out_dir, fname)

        if args.ext in ("jpg", "jpeg"):
            im.save(out_path, quality=95)
        else:
            im.save(out_path)

        saved += 1


if __name__ == "__main__":
    main()

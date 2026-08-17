#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import cv2
from skimage.color import rgb2hed
from tqdm import tqdm


def pick_array(npz):
    if "arr_0" in npz.files:
        return npz["arr_0"], "arr_0"
    if len(npz.files) == 0:
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


def is_tile_valid(tile_rgb, max_bg, max_blood, min_nuclei):
    hsv = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    bg_mask = (v > 210) & (s < 30)
    bg_ratio = np.sum(bg_mask) / bg_mask.size
    if bg_ratio > max_bg:
        return False, "Scartato: Troppo Background/Grasso ({:.1f}%)".format(bg_ratio * 100)

    blood_mask = ((h < 15) | (h > 165)) & (s > 100) & (v > 50)
    blood_ratio = np.sum(blood_mask) / blood_mask.size
    if blood_ratio > max_blood:
        return False, "Scartato: Troppa Emorragia ({:.1f}%)".format(blood_ratio * 100)

    hed = rgb2hed(tile_rgb)
    h_channel = hed[:, :, 0] 
    
    nuclei_mask = h_channel > 0.05 
    nuclei_ratio = np.sum(nuclei_mask) / nuclei_mask.size
    if nuclei_ratio < min_nuclei:
        return False, "Scartato: Povero di nuclei/Necrosi ({:.1f}%)".format(nuclei_ratio * 100)

    return True, "Valido"

def process_npz_file(file_path, output_path, args):
    try:
        with np.load(file_path, allow_pickle=True) as npz:
            data, key = pick_array(npz)

        if not (isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 1):
            print(f"\n[SKIP] Formato inatteso in {file_path}. Mi aspetto array 2D.")
            return 0, 0, {}
        
        valid_rows = []
        stats = {"Valido": 0, "Background": 0, "Emorragia": 0, "Necrosi": 0}
        
        total_tiles = data.shape[0]

        for i in range(total_tiles):
            raw_img = data[i, 0]
            
            img_rgb = to_uint8_rgb(raw_img, is_bgr=args.bgr)
            
            is_valid, reason = is_tile_valid(img_rgb, args.max_bg, args.max_blood, args.min_nuclei)
            
            if is_valid:
                valid_rows.append(data[i])
                stats["Valido"] += 1
            else:
                if "Background" in reason: stats["Background"] += 1
                elif "Emorragia" in reason: stats["Emorragia"] += 1
                elif "Necrosi" in reason: stats["Necrosi"] += 1

        if len(valid_rows) > 0:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            valid_data_array = np.array(valid_rows, dtype=data.dtype)
            np.savez_compressed(output_path, arr_0=valid_data_array)
        
        return total_tiles, len(valid_rows), stats
        
    except Exception as e:
        print(f"\nErrore nell'elaborazione di {file_path}: {e}")
        return 0, 0, {}

def main():
    parser = argparse.ArgumentParser(description="Filtra tile H&E rimuovendo sangue, background, grasso e necrosi.")
    parser.add_argument("--input_dir", type=str, default="/home/sg510849/giuSpathis/data/heFiltered", help="Cartella input")
    parser.add_argument("--output_dir", type=str, default="/home/sg510849/giuSpathis/data/heFiltered_clean", help="Cartella output")
    parser.add_argument("--bgr", action="store_true", help="Inverte i canali da BGR a RGB prima del filtro")
    
    parser.add_argument("--max_bg", type=float, default=0.45, help="Max background/grasso tollerato (default: 0.45)")
    parser.add_argument("--max_blood", type=float, default=0.20, help="Max sangue tollerato (default: 0.20)")
    parser.add_argument("--min_nuclei", type=float, default=0.03, help="Min nuclei per scartare necrosi (default: 0.03)")
    
    args = parser.parse_args()


    npz_files = glob.glob(os.path.join(args.input_dir, '**', '*.npz'), recursive=True)
    
    total_original = 0
    total_kept = 0
    
    for file_path in tqdm(npz_files, desc="Elaborazione file NPZ"):
        rel_path = os.path.relpath(file_path, args.input_dir)
        out_path = os.path.join(args.output_dir, rel_path)
        
        orig_count, kept_count, stats = process_npz_file(file_path, out_path, args)
        
        total_original += orig_count
        total_kept += kept_count
        
        if orig_count > 0:
            tqdm.write(f"[{rel_path}] Letti: {orig_count} | Tenuti: {kept_count} | Scarti -> BG/Grasso: {stats.get('Background', 0)}, Sangue: {stats.get('Emorragia', 0)}, Necrosi: {stats.get('Necrosi', 0)}")


if __name__ == "__main__":
    main()
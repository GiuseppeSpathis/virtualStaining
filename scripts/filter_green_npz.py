#!/usr/bin/env python3
import os
import argparse
import numpy as np
from tqdm.auto import tqdm

def parse_args():
    p = argparse.ArgumentParser(description="Filtra le tiles NPZ verdi e bilancia il dataset.")
    p.add_argument("--input_dir", type=str, default=os.path.expanduser("~/giuSpathis/data/ihc"),
                   help="Cartella principale contenente i file .npz.")
    p.add_argument("--output_dir", type=str, default=os.path.expanduser("~/giuSpathis/data/ihc_filtered"),
                   help="Cartella dove salvare i file .npz filtrati.")
    p.add_argument("--key", type=str, default="arr_0",
                   help="Chiave dell'array all'interno del file .npz.")
    p.add_argument("--green_margin", type=int, default=20,
                   help="Margine per considerare il canale G dominante su R e B.")
    p.add_argument("--max_green_ratio", type=float, default=0.02,
                   help="Percentuale massima tollerata di pixel verdi per tile (es. 0.02 per 2%).")
    p.add_argument("--max_tiles_per_file", type=int, default=600,
                   help="Numero MASSIMO di tiles da tenere per file (0 = nessun limite).")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed per il campionamento casuale in modo da renderlo riproducibile.")
    return p.parse_args()

def process_and_filter_npz(filepath, out_filepath, args):
    data = np.load(filepath, allow_pickle=True, mmap_mode='r')
    
    if args.key not in data:
        return 0, 0
        
    tiles = data[args.key]
    n_original = tiles.shape[0]
    
    if n_original == 0:
        return 0, 0

    valid_indices = []
    batch_size = 500
    
    for start in range(0, n_original, batch_size):
        end = min(start + batch_size, n_original)
        batch = tiles[start:end]
        
        if batch.ndim == 2 and batch.shape[1] >= 1:
            images = np.stack(batch[:, 0])
        else:
            images = batch
            
        R = images[..., 0].astype(np.int16)
        G = images[..., 1].astype(np.int16)
        B = images[..., 2].astype(np.int16)
        
        green_mask = (G > R + args.green_margin) & (G > B + args.green_margin)
        green_pixels_per_tile = np.sum(green_mask, axis=(1, 2))
        
        pixels_total = images.shape[1] * images.shape[2]
        green_ratio_per_tile = green_pixels_per_tile / pixels_total
        
        local_valid = np.where(green_ratio_per_tile <= args.max_green_ratio)[0]
        valid_indices.extend(local_valid + start)

    n_filtered = len(valid_indices)
    
    if args.max_tiles_per_file > 0 and n_filtered > args.max_tiles_per_file:
        np.random.seed(args.seed) 
        valid_indices = np.random.choice(valid_indices, size=args.max_tiles_per_file, replace=False)
        valid_indices = np.sort(valid_indices)
        n_filtered = len(valid_indices)

    if n_filtered > 0:
        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        filtered_tiles = tiles[valid_indices]
        
        save_dict = {args.key: filtered_tiles}
        np.savez_compressed(out_filepath, **save_dict)
        
    return n_original, n_filtered

def main():
    args = parse_args()
    

    npz_files = []
    for root, _, files in os.walk(args.input_dir):
        for fn in files:
            if fn.endswith('.npz'):
                npz_files.append(os.path.join(root, fn))

    total_orig = 0
    total_filt = 0

    for path in tqdm(npz_files, desc="Processando file"):
        rel_path = os.path.relpath(path, args.input_dir)
        out_path = os.path.join(args.output_dir, rel_path)
        
        n_orig, n_filt = process_and_filter_npz(path, out_path, args)
        
        total_orig += n_orig
        total_filt += n_filt
        
        if n_orig > 0:
            dropped_green = (n_orig - n_filt) if n_filt >= args.max_tiles_per_file else (n_orig - n_filt)
            print(f"{rel_path}: Salvate {n_filt}/{n_orig} tiles")


if __name__ == "__main__":
    main()

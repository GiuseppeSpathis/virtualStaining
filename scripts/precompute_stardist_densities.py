#!/usr/bin/env python3

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from stardist.models import StarDist2D

def parse_args():
    p = argparse.ArgumentParser(description="Pre-computa la densità dei nuclei su griglia 8x8 per il virtual staining")
    p.add_argument("--he_dir", type=str, required=True, help="Cartella contenente i file .npz H&E o le sottocartelle")
    p.add_argument("--output_dir", type=str, required=True, help="Cartella root dove salvare i tensori di densità .npy")
    p.add_argument("--key", type=str, default="arr_0")
    p.add_argument("--grid_size", type=int, default=8, help="Dimensione della griglia (8 = 64 token)")
    
    # Parametri StarDist
    p.add_argument("--stardist_model", type=str, default="2D_versatile_he")
    p.add_argument("--stardist_prob_thresh", type=float, default=0.692478)
    p.add_argument("--stardist_nms_thresh", type=float, default=0.3)
    p.add_argument("--stardist_min_area", type=int, default=20)
    
    return p.parse_args()

def remove_small_instances(mask_np: np.ndarray, min_area: int) -> np.ndarray:
    if mask_np.max() == 0: return np.zeros_like(mask_np, dtype=np.int32)
    out = np.zeros_like(mask_np, dtype=np.int32)
    ids = np.unique(mask_np)
    ids = ids[ids > 0]
    new_id = 1
    for idx in ids:
        area = int((mask_np == idx).sum())
        if area >= min_area:
            out[mask_np == idx] = new_id
            new_id += 1
    return out

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("[INFO] Caricamento StarDist...")
    stardist_model = StarDist2D.from_pretrained(args.stardist_model)
    
    # Cerca le sottocartelle (subtype)
    subtypes = [d for d in os.listdir(args.he_dir) if os.path.isdir(os.path.join(args.he_dir, d))]
    
    # Se non ci sono sottocartelle, processa direttamente la root
    if not subtypes:
        subtypes = ["."]

    total_files_processed = 0

    for subtype in subtypes:
        src_subdir = os.path.join(args.he_dir, subtype) if subtype != "." else args.he_dir
        tgt_subdir = os.path.join(args.output_dir, subtype) if subtype != "." else args.output_dir
        
        # Crea la sottocartella di output speculare (es. stardist_densities_8x8/ccRCC)
        os.makedirs(tgt_subdir, exist_ok=True)
        
        files = [f for f in os.listdir(src_subdir) if f.endswith(".npz")]
        files.sort()
        if not files:
            continue
            
        print(f"\n[INFO] Trovati {len(files)} file in {subtype}")
        
        for filename in tqdm(files, desc=f"Processando {subtype}"):
            input_path = os.path.join(src_subdir, filename)
            
            # Carica il file NPZ
            data = np.load(input_path, allow_pickle=True)
            arr = data[args.key]
            
            N = arr.shape[0]
            # Creiamo un array vuoto per contenere le mappe di densità di tutto il file (N, 8, 8)
            densities_array = np.zeros((N, args.grid_size, args.grid_size), dtype=np.float32)
            
            for i in range(N):
                tile = arr[i, 0]
                
                # Stessa logica di resize che avevi nel training
                if isinstance(tile, np.ndarray) and tile.shape == (256, 256, 3):
                    tile = np.array(Image.fromarray(tile).resize((1024, 1024), Image.BICUBIC))
                    
                img_f = tile.astype(np.float32) / 255.0
                
                # 1. Inferenza StarDist
                labels, _ = stardist_model.predict_instances(
                    img_f, axes="YXC", prob_thresh=args.stardist_prob_thresh, nms_thresh=args.stardist_nms_thresh
                )
                labels = remove_small_instances(labels.astype(np.int32), min_area=args.stardist_min_area)
                
                # 2. Maschera binaria -> Float
                bin_np = (labels > 0).astype(np.float32)
                
                # 3. Pooling su griglia 8x8 con PyTorch
                bt = torch.from_numpy(bin_np).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
                d8 = F.adaptive_avg_pool2d(bt, (args.grid_size, args.grid_size)) # (1, 1, 8, 8)
                
                densities_array[i] = d8.squeeze().numpy()
                
            data.close() # Chiudiamo il file caricato
                
            # Salva l'array di densità
            out_filename = filename.replace(".npz", "_densities.npy")
            out_path = os.path.join(tgt_subdir, out_filename)
            np.save(out_path, densities_array)
            total_files_processed += 1

    print(f"\n[INFO] Finito! {total_files_processed} file salvati in totale.")

if __name__ == "__main__":
    main()
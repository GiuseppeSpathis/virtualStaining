import os
import numpy as np
from skimage.color import rgb2hed
from tqdm import tqdm

BASE_DIR = "/home/sg510849/giuSpathis/data"
CLASSES_TO_FILTER = ["ccRCC", "onco"]
DAB_PERCENTAGE_LIMIT = 5.0  
DAB_INTENSITY_THRESH = 0.05 

DIR_IHC = "ihc_filtered"
KEY_NPZ = "arr_0"

def calculate_dab_percentage(tile_rgb):
    tile_rgb = tile_rgb.astype(np.float32) / 255.0
        
    hed = rgb2hed(tile_rgb)
    dab_channel = hed[:, :, 2] 
    
    dab_pixels = (dab_channel > DAB_INTENSITY_THRESH).sum()
    total_pixels = dab_channel.size
    
    return (dab_pixels / total_pixels) * 100

def main():
    
    report = []

    for cls in CLASSES_TO_FILTER:
        
        path_ihc = os.path.join(BASE_DIR, DIR_IHC, cls)
        
        if not os.path.exists(path_ihc):
            continue
            
        files_ihc = sorted([f for f in os.listdir(path_ihc) if f.endswith('.npz')])

        for f_ihc in files_ihc:
            ihc_data = np.load(os.path.join(path_ihc, f_ihc), allow_pickle=True)[KEY_NPZ]
            original_tiles_count = ihc_data.shape[0]
            
            keep_indices = []
            
            for idx in tqdm(range(original_tiles_count), leave=False):
                tile_img = ihc_data[idx, 0]
                
                tile_img = np.asarray(tile_img).astype(np.uint8)
                
                if tile_img.ndim == 3 and tile_img.shape[0] in [3, 4]:
                    tile_img = np.transpose(tile_img, (1, 2, 0))
                
                if tile_img.shape[-1] == 4:
                    tile_img = tile_img[..., :3]
                
                if tile_img.shape[-1] != 3:

                dab_percent = calculate_dab_percentage(tile_img)
                
                if dab_percent <= DAB_PERCENTAGE_LIMIT:
                    keep_indices.append(idx)
            
            new_tiles_count = len(keep_indices)
            
            filtered_ihc_data = ihc_data[keep_indices]
            np.savez(os.path.join(path_ihc, f_ihc), **{KEY_NPZ: filtered_ihc_data})
            
            report_str = f"[{cls}] {f_ihc} -> Rimaste: {new_tiles_count}/{original_tiles_count} (Scartate: {original_tiles_count - new_tiles_count})"
            report.append(report_str)
            print(f"  -> Salvato. Rimaste: {new_tiles_count}/{original_tiles_count}")

    for r in report:
        print(r)

if __name__ == "__main__":
    main()
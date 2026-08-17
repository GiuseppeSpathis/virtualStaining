import os
import numpy as np
from tqdm import tqdm
from stardist.models import StarDist2D
from csbdeep.utils import normalize

BASE_DIR = "/home/sg510849/giuSpathis/data/he"  
OUT_DIR = "/home/sg510849/giuSpathis/data/heFiltered"  
CLASSES_TO_FILTER = ["Annotations_chromo", "Annotations_onco", "ccRCC", "pRCC"]

MIN_NUCLEI_COUNT = 15  

KEY_NPZ = "arr_0"

def main():
    model = StarDist2D.from_pretrained('2D_versatile_he')
    
    
    report = []

    for cls in CLASSES_TO_FILTER:
        path_data = os.path.join(BASE_DIR, cls)
        
        out_path_cls = os.path.join(OUT_DIR, cls)
        
        if not os.path.exists(path_data):
            continue
            
        os.makedirs(out_path_cls, exist_ok=True)
            
        files_data = sorted([f for f in os.listdir(path_data) if f.endswith('.npz')])

        
        for f_data in files_data:
            data_path_full = os.path.join(path_data, f_data)
            data_npz = np.load(data_path_full, allow_pickle=True)[KEY_NPZ]
            
            original_tiles_count = data_npz.shape[0]
            keep_indices = []
            
            
            for idx in tqdm(range(original_tiles_count), leave=False):
                try:
                    tile_img = data_npz[idx, 0]
                except (IndexError, TypeError):
                    tile_img = data_npz[idx]
                
                if isinstance(tile_img, (list, tuple)) or (isinstance(tile_img, np.ndarray) and tile_img.dtype == object):
                    tile_img = tile_img[0]
                    
                tile_img = np.asarray(tile_img).astype(np.uint8)
                
                if tile_img.ndim == 3 and tile_img.shape[0] in [3, 4]:
                    tile_img = np.transpose(tile_img, (1, 2, 0))
                
                if tile_img.shape[-1] == 4:
                    tile_img = tile_img[..., :3]

                img_norm = normalize(tile_img, 1, 99.8, axis=(0, 1))
                labels, _ = model.predict_instances(img_norm, verbose=False)
                
                nuclei_count = labels.max()
                
                if nuclei_count >= MIN_NUCLEI_COUNT:
                    keep_indices.append(idx)
            
            new_tiles_count = len(keep_indices)
            discarded_count = original_tiles_count - new_tiles_count
            
            filtered_data = data_npz[keep_indices]
            
            out_file_full = os.path.join(out_path_cls, f_data)
            
            np.savez(out_file_full, **{KEY_NPZ: filtered_data})
            
            report_str = f"[{cls}] {f_data}: Scartate {discarded_count} tiles (Rimaste: {new_tiles_count}/{original_tiles_count})"
            report.append(report_str)

    if not report:
    else:
        for r in report:

if __name__ == "__main__":
    main()
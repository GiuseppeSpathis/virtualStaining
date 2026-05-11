import os
import numpy as np
from skimage.color import rgb2hed
from tqdm import tqdm

# --- CONFIGURAZIONE ---
BASE_DIR = "/home/sg510849/giuSpathis/data"
CLASSES_TO_FILTER = ["ccRCC", "onco"]
DAB_PERCENTAGE_LIMIT = 5.0  # Soglia massima del 5% di pixel DAB
DAB_INTENSITY_THRESH = 0.05 # Soglia di intensità sul canale DAB (> 0.05 è considerato marrone)

DIR_IHC = "ihc_filtered"
KEY_NPZ = "arr_0"

def calculate_dab_percentage(tile_rgb):
    """
    Converte un'immagine RGB nello spazio HED e calcola la % di pixel DAB.
    """
    # Converte in float [0, 1] per scikit-image
    tile_rgb = tile_rgb.astype(np.float32) / 255.0
        
    hed = rgb2hed(tile_rgb)
    dab_channel = hed[:, :, 2] # Il canale 2 è il DAB (Marrone)
    
    # Maschera i pixel che hanno una componente DAB superiore alla soglia
    dab_pixels = (dab_channel > DAB_INTENSITY_THRESH).sum()
    total_pixels = dab_channel.size
    
    return (dab_pixels / total_pixels) * 100

def main():
    print(f"Inizio filtraggio DAB (> {DAB_PERCENTAGE_LIMIT}%) sulle classi: {CLASSES_TO_FILTER}")
    
    report = []

    for cls in CLASSES_TO_FILTER:
        print(f"\n--- Processando la classe: {cls} ---")
        
        path_ihc = os.path.join(BASE_DIR, DIR_IHC, cls)
        
        # Se la cartella non esiste, salta (utile per evitare errori)
        if not os.path.exists(path_ihc):
            print(f"Cartella {path_ihc} non trovata. Salto.")
            continue
            
        files_ihc = sorted([f for f in os.listdir(path_ihc) if f.endswith('.npz')])

        for f_ihc in files_ihc:
            ihc_data = np.load(os.path.join(path_ihc, f_ihc), allow_pickle=True)[KEY_NPZ]
            original_tiles_count = ihc_data.shape[0]
            
            keep_indices = []
            print(f"Analizzando {f_ihc} ({original_tiles_count} tiles)...")
            
            for idx in tqdm(range(original_tiles_count), leave=False):
                # Estrazione diretta dell'immagine
                tile_img = ihc_data[idx, 0]
                
                # Assicuriamoci che sia un numpy array uint8
                tile_img = np.asarray(tile_img).astype(np.uint8)
                
                # Traspone se è in formato (Canali, H, W)
                if tile_img.ndim == 3 and tile_img.shape[0] in [3, 4]:
                    tile_img = np.transpose(tile_img, (1, 2, 0))
                
                # Rimuove il canale Alpha se presente (RGBA -> RGB)
                if tile_img.shape[-1] == 4:
                    tile_img = tile_img[..., :3]
                
                if tile_img.shape[-1] != 3:
                    raise ValueError(f"Dimensione canali errata dopo il processing: {tile_img.shape}")

                dab_percent = calculate_dab_percentage(tile_img)
                
                if dab_percent <= DAB_PERCENTAGE_LIMIT:
                    keep_indices.append(idx)
            
            new_tiles_count = len(keep_indices)
            
            # Applica il filtro e salva sovrascrivendo il file IHC
            filtered_ihc_data = ihc_data[keep_indices]
            np.savez(os.path.join(path_ihc, f_ihc), **{KEY_NPZ: filtered_ihc_data})
            
            report_str = f"[{cls}] {f_ihc} -> Rimaste: {new_tiles_count}/{original_tiles_count} (Scartate: {original_tiles_count - new_tiles_count})"
            report.append(report_str)
            print(f"  -> Salvato. Rimaste: {new_tiles_count}/{original_tiles_count}")

    print("\n================ REPORT FINALE ================")
    for r in report:
        print(r)
    print("===============================================\n")

if __name__ == "__main__":
    main()
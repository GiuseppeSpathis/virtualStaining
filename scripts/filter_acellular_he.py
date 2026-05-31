import os
import numpy as np
from tqdm import tqdm
from stardist.models import StarDist2D
from csbdeep.utils import normalize

# --- CONFIGURAZIONE ---
BASE_DIR = "/home/sg510849/giuSpathis/data/he"  # Cartella di origine
OUT_DIR = "/home/sg510849/giuSpathis/data/heFiltered"  # NUOVA cartella di destinazione
CLASSES_TO_FILTER = ["Annotations_chromo", "Annotations_onco", "ccRCC", "pRCC"]

# SOGLIA CELLULARE: Quanti nuclei minimi deve avere una tile per non essere considerata "vuota" o "sangue"?
# Su una tile 1024x1024 a 20x, 15-20 nuclei è un limite ragionevole per scartare i detriti.
MIN_NUCLEI_COUNT = 15  

KEY_NPZ = "arr_0"

def main():
    print("Inizializzazione modello StarDist (2D_versatile_he)...")
    # Carica il modello pre-addestrato standard per istologia H&E
    model = StarDist2D.from_pretrained('2D_versatile_he')
    
    print(f"\nInizio filtraggio live (Nuclei Minimi >= {MIN_NUCLEI_COUNT})")
    print(f"I file filtrati verranno salvati in: {OUT_DIR}\n")
    
    report = []

    for cls in CLASSES_TO_FILTER:
        path_data = os.path.join(BASE_DIR, cls)
        
        # Percorso della nuova sottocartella di destinazione (es. .../heFiltered/ccRCC)
        out_path_cls = os.path.join(OUT_DIR, cls)
        
        if not os.path.exists(path_data):
            print(f"  [!] Cartella di origine {path_data} non trovata. Salto.")
            continue
            
        # Crea la cartella di destinazione se non esiste
        os.makedirs(out_path_cls, exist_ok=True)
            
        files_data = sorted([f for f in os.listdir(path_data) if f.endswith('.npz')])

        print(f"\n--- Processando la classe: {cls} ---")
        
        for f_data in files_data:
            data_path_full = os.path.join(path_data, f_data)
            data_npz = np.load(data_path_full, allow_pickle=True)[KEY_NPZ]
            
            original_tiles_count = data_npz.shape[0]
            keep_indices = []
            
            print(f"Analizzando {f_data} ({original_tiles_count} tiles)...")
            
            for idx in tqdm(range(original_tiles_count), leave=False):
                # --- Estrazione Robusta ---
                try:
                    # Tenta l'estrazione [indice_tile, 0] come nel tuo script originale
                    tile_img = data_npz[idx, 0]
                except (IndexError, TypeError):
                    # Se fallisce, prova a prendere tutto l'elemento
                    tile_img = data_npz[idx]
                
                # Sicurezza aggiuntiva: se è ancora un object array o una tupla, prendi il primo elemento
                if isinstance(tile_img, (list, tuple)) or (isinstance(tile_img, np.ndarray) and tile_img.dtype == object):
                    tile_img = tile_img[0]
                    
                tile_img = np.asarray(tile_img).astype(np.uint8)
                
                # Traspone se è in formato (Canali, H, W)
                if tile_img.ndim == 3 and tile_img.shape[0] in [3, 4]:
                    tile_img = np.transpose(tile_img, (1, 2, 0))
                
                # Rimuove l'Alpha channel se presente
                if tile_img.shape[-1] == 4:
                    tile_img = tile_img[..., :3]

                # --- INFERENZA STARDIST ---
                img_norm = normalize(tile_img, 1, 99.8, axis=(0, 1))
                labels, _ = model.predict_instances(img_norm, verbose=False)
                
                nuclei_count = labels.max()
                
                if nuclei_count >= MIN_NUCLEI_COUNT:
                    keep_indices.append(idx)
            
            new_tiles_count = len(keep_indices)
            discarded_count = original_tiles_count - new_tiles_count
            
            # Applica il filtro
            filtered_data = data_npz[keep_indices]
            
            # Costruisci il nuovo percorso di salvataggio
            out_file_full = os.path.join(out_path_cls, f_data)
            
            # Salva il nuovo file
            np.savez(out_file_full, **{KEY_NPZ: filtered_data})
            
            report_str = f"[{cls}] {f_data}: Scartate {discarded_count} tiles (Rimaste: {new_tiles_count}/{original_tiles_count})"
            report.append(report_str)
            print(f"  -> Salvato in heFiltered. Rimaste: {new_tiles_count}/{original_tiles_count}")

    print("\n================ REPORT FINALE ================")
    if not report:
        print("Nessuna elaborazione effettuata.")
    else:
        for r in report:
            print(r)
    print("===============================================\n")

if __name__ == "__main__":
    main()
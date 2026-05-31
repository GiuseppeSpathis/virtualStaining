import os
import numpy as np
import torch
from torch.utils.data import Dataset


class NPZFolderDataset(Dataset):
    """
    Dataset paired H&E / IHC da cartelle con sottocartelle per subtype:
        folder_path/
            ccRCC/
            chRCC/
            onco/
            pRCC/
        mask_folder_path/
            ccRCC/
            chRCC/
            onco/
            pRCC/

    Matching:
      - per ogni subtype
      - ordina i file .npz di H&E
      - ordina i file .npz di IHC
      - accoppia per posizione: primo con primo, secondo con secondo, ecc.

    Ogni sample restituisce:
      Se precomputed_path NON è fornito o il file non esiste: (img, mask)
      Se precomputed_path è fornito e il file esiste: (img, mask, density_map_8x8)
    """

    def __init__(
        self,
        folder_path: str,
        mask_folder_path: str = None,
        precomputed_path: str = None, # Directory con le densità (es. stardist_densities_8x8)
        key: str = "arr_0",
        max_per_file: int = 300,
        sort_files: bool = True,
        verbose: bool = True,
    ):
        self.folder_path = os.path.expanduser(folder_path)
        self.mask_folder_path = os.path.expanduser(mask_folder_path) if mask_folder_path else None
        self.precomputed_path = os.path.expanduser(precomputed_path) if precomputed_path else None
        self.key = key
        self.max_per_file = max_per_file
        self.verbose = verbose

        self.samples = []

        if self.verbose:
            mode = "Immagini + Paired-by-order" if self.mask_folder_path else "Immagini"
            print(f"Indicizzazione {mode} (max {max_per_file} tile per file)...")

        # Sottocartelle subtype presenti in folder_path (può essere HE o IHC a seconda dello script)
        # Se non ci sono sottocartelle, usiamo la root
        subtypes = [
            d for d in os.listdir(self.folder_path)
            if os.path.isdir(os.path.join(self.folder_path, d))
        ]
        
        # Gestione fallback se la cartella contiene direttamente i file .npz (senza sottocartelle subtype)
        if not subtypes:
            subtypes = ["."]

        if sort_files and subtypes != ["."]:
            subtypes.sort()

        for subtype in subtypes:
            src_subdir = self.folder_path if subtype == "." else os.path.join(self.folder_path, subtype)

            src_files = [
                f for f in os.listdir(src_subdir)
                if f.endswith(".npz")
                # Abbiamo rimosso il filtro f.startswith("he_") così funziona anche per "ihc_"
            ]
            if sort_files:
                src_files.sort()

            # Se c'è una cartella target accoppiata
            if self.mask_folder_path:
                tgt_subdir = self.mask_folder_path if subtype == "." else os.path.join(self.mask_folder_path, subtype)

                if not os.path.isdir(tgt_subdir):
                    if self.verbose:
                        print(f"Salto subtype {subtype}: cartella target/IHC non trovata")
                    continue

                tgt_files = [
                    f for f in os.listdir(tgt_subdir)
                    if f.endswith(".npz")
                ]
                if sort_files:
                    tgt_files.sort()

                n_pairs = min(len(src_files), len(tgt_files))

                if self.verbose:
                    print(
                        f"[{subtype}] Source files: {len(src_files)} | "
                        f"Target files: {len(tgt_files)} | paired: {n_pairs}"
                    )

                for j in range(n_pairs):
                    src_fp = os.path.join(src_subdir, src_files[j])
                    tgt_fp = os.path.join(tgt_subdir, tgt_files[j])

                    try:
                        src_data = np.load(src_fp, mmap_mode="r", allow_pickle=True)
                        src_arr = src_data[self.key]
                        n_src = int(src_arr.shape[0])
                        src_data.close()

                        tgt_data = np.load(tgt_fp, mmap_mode="r", allow_pickle=True)
                        tgt_arr = tgt_data[self.key]
                        n_tgt = int(tgt_arr.shape[0])
                        tgt_data.close()

                        take = min(n_src, n_tgt, self.max_per_file)

                        if self.verbose:
                            print(
                                f"  pair {j+1}: "
                                f"{os.path.basename(src_fp)} <-> {os.path.basename(tgt_fp)} "
                                f"| SRC={n_src} TGT={n_tgt} use={take}"
                            )

                        for i in range(take):
                            self.samples.append((src_fp, tgt_fp, i))

                    except Exception as e:
                        if self.verbose:
                            print(
                                f"Errore nell'indicizzazione pair "
                                f"{os.path.basename(src_fp)} <-> {os.path.basename(tgt_fp)}: {e}"
                            )

            else:
                # Modalità non paired (es. Training LoRA solo su IHC)
                for f in src_files:
                    src_fp = os.path.join(src_subdir, f)
                    try:
                        data = np.load(src_fp, mmap_mode="r", allow_pickle=True)
                        arr = data[self.key]
                        n = int(arr.shape[0])
                        data.close()
                        
                        take = min(n, self.max_per_file)
                        for i in range(take):
                            self.samples.append((src_fp, None, i))
                    except Exception as e:
                        if self.verbose:
                            print(f"Errore nell'indicizzazione di {src_fp}: {e}")

        if self.verbose:
            print(f"Dataset pronto: {len(self.samples)} campioni totali.")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _tile_to_uint8_numpy(tile):
        if not isinstance(tile, np.ndarray):
            tile = np.array(tile)

        if tile.dtype == object:
            tile = np.array(tile.tolist(), dtype=np.uint8)
        elif tile.dtype != np.uint8:
            tile = tile.astype(np.uint8)

        return tile

    def __getitem__(self, idx):
        img_path, mask_path, internal_idx = self.samples[idx]

        # --- Immagine Source (Es. H&E o IHC isolata) ---
        data_src = np.load(img_path, mmap_mode="r", allow_pickle=True)
        arr_src = data_src[self.key]
        
        tile_img = self._tile_to_uint8_numpy(arr_src[internal_idx, 0])
        # Lasciamo img in formato H,W,C in range [0, 255] uint8 per compatibilità col nuovo estensore Fast
        img = tile_img
        data_src.close()

        # --- Immagine Target (Es. IHC) ---
        if mask_path is not None:
            data_tgt = np.load(mask_path, mmap_mode="r", allow_pickle=True)
            arr_tgt = data_tgt[self.key]
            
            tile_mask = self._tile_to_uint8_numpy(arr_tgt[internal_idx, 0])
            mask = tile_mask
            data_tgt.close()
        else:
            mask = img

        # --- Densità Pre-computate (Se Esistono) ---
        # Formattiamo il path: sostituiamo la folder source con la folder precomputed
        # e cambiamo estensione da .npz a _densities.npy
        if self.precomputed_path is not None:
            # Ricostruiamo il percorso
            rel_path = os.path.relpath(img_path, self.folder_path)
            density_file = rel_path.replace(".npz", "_densities.npy")
            density_full_path = os.path.join(self.precomputed_path, density_file)

            if os.path.exists(density_full_path):
                # Carichiamo solo l'indice specifico senza portare tutto in RAM
                try:
                    densities_array = np.load(density_full_path, mmap_mode="r")
                    density_map_8x8 = torch.from_numpy(np.array(densities_array[internal_idx]))
                    return img, mask, density_map_8x8
                except Exception:
                    pass # Se c'è un errore di IO, fallback al return a 2 elementi

        # Fallback standard
        return img, mask
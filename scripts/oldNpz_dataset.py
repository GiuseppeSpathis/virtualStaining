import os
import numpy as np
import torch
from torch.utils.data import Dataset


class NPZFolderDataset(Dataset):

    def __init__(
        self,
        folder_path: str,
        mask_folder_path: str = None,  
        precomputed_path=None,
        key: str = "arr_0",
        max_per_file: int = 300,
        sort_files: bool = True,
        verbose: bool = True,
    ):
        self.folder_path = os.path.expanduser(folder_path)
        self.precomputed_path = os.path.expanduser(precomputed_path) if precomputed_path else None
        self.mask_folder_path = os.path.expanduser(mask_folder_path) if mask_folder_path else None
        self.key = key
        self.max_per_file = max_per_file
        self.verbose = verbose

        files = []
        for root, _, filenames in os.walk(self.folder_path):
            for filename in filenames:
                if filename.endswith(".npz"):
                    rel_path = os.path.relpath(os.path.join(root, filename), self.folder_path)
                    files.append(rel_path)
                    
        if sort_files:
            files.sort()

        self.samples = []

        for f in files:
            img_fp = os.path.join(self.folder_path, f)
            mask_fp = os.path.join(self.mask_folder_path, f) if self.mask_folder_path else None
            
            if self.mask_folder_path and not os.path.exists(mask_fp):
                continue

            try:
                data = np.load(img_fp, mmap_mode="r", allow_pickle=True)
                arr = data[self.key]
                n = int(arr.shape[0])
                take = min(n, max_per_file)
                
                if mask_fp:
                    mask_data = np.load(mask_fp, mmap_mode="r", allow_pickle=True)
                    n_masks = int(mask_data[self.key].shape[0])
                    take = min(take, n_masks)

                for i in range(take):
                    self.samples.append((img_fp, mask_fp, i))
            except Exception as e:


        self._cache = None

    def __len__(self):
        return len(self.samples)

    def _init_cache_if_needed(self):
        if self._cache is None:
            self._cache = {
                "npz": {},  # file_path -> np.load(...) object
                "arr": {},  # file_path -> reference a data[key]
            }

    def _get_arr(self, file_path: str):
        self._init_cache_if_needed()

        if file_path in self._cache["arr"]:
            return self._cache["arr"][file_path]

        data = np.load(file_path, mmap_mode="r", allow_pickle=True)
        arr = data[self.key]

        self._cache["npz"][file_path] = data
        self._cache["arr"][file_path] = arr
        return arr

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

        if mask_path:
            mask_arr = self._get_arr(mask_path)
            tile_mask = self._tile_to_uint8_numpy(mask_arr[internal_idx, 0])
            mask = torch.from_numpy(tile_mask).float() / 255.0

            if mask.ndim == 2:
                mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)
            elif mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
        else:
            mask = None 

        if self.precomputed_path:
            pt_file = os.path.join(self.precomputed_path, f"tile_{idx}.pt")
            
            if os.path.exists(pt_file):
                data = torch.load(pt_file, map_location="cpu")
                latents = data["latents"].squeeze(0) 
                uni_emb = data["uni_emb"].squeeze(0)
                
                return latents, uni_emb, mask

        img_arr = self._get_arr(img_path)
        tile_img = self._tile_to_uint8_numpy(img_arr[internal_idx, 0])
        img = torch.from_numpy(tile_img).float() / 255.0
        if img.ndim == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)

        if mask is None:
            mask = img

        return img, mask
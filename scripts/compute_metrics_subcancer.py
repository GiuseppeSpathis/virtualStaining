#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any

import pickle
class Numpy2Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if 'numpy._core' in module:
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)

_orig_load = pickle.load
def custom_load(file, **kwargs):
    return Numpy2Unpickler(file, **kwargs).load()
pickle.load = custom_load

import sys
import numpy as np

import torch
import pandas as pd
from scipy.stats import wasserstein_distance
from skimage.feature import canny
from skimage.color import rgb2gray, rgb2hed
from skimage.morphology import disk, binary_dilation  

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

try:
    from cellpose import models
    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False

try:
    from tiatoolbox.models.engine.nucleus_instance_segmentor import NucleusInstanceSegmentor
    HOVERNET_AVAILABLE = True
except ImportError:
    HOVERNET_AVAILABLE = False

try:
    from cellseg_models_pytorch.models import CPPNet
    CPPNET_AVAILABLE = True
except Exception as e:
    CPPNET_AVAILABLE = False

try:
    from stardist.models import StarDist2D
    STARDIST_AVAILABLE = True
except ImportError:
    STARDIST_AVAILABLE = False

try:
    from splinedist.models import SplineDist2D
    SPLINEDIST_AVAILABLE = True
except ImportError:
    SPLINEDIST_AVAILABLE = False


try:
    import tensorflow as tf
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
except Exception as e:

NPZ_EXTS = {".npz"}

def _tile_to_uint8_numpy(tile) -> np.ndarray:
    if not isinstance(tile, np.ndarray):
        tile = np.array(tile)
    if tile.dtype == object:
        tile = np.array(tile.tolist(), dtype=np.uint8)
    elif tile.dtype != np.uint8:
        tile = tile.astype(np.uint8)
    return tile

def _ensure_hwc_rgb_uint8(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = np.stack([x, x, x], axis=-1)
    elif x.ndim == 3:
        if x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
            x = np.transpose(x, (1, 2, 0))
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
    if x.dtype != np.uint8:
        x = x.astype(np.float32)
        mx = float(np.nanmax(x)) if x.size else 0.0
        if mx <= 1.5:
            x = x * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x

def list_npz_basenames(folder: str) -> List[str]:
    p = Path(os.path.expanduser(folder))
    if not p.exists(): return []
    return sorted([Path(x).name for x in p.iterdir() if x.is_file() and x.suffix.lower() in NPZ_EXTS])

def load_all_npz_tiles(npz_path: str, key: str, max_tiles: int = None) -> List[np.ndarray]:
    if np.__version__.startswith("1."):
        if 'numpy._core' not in sys.modules:
            sys.modules['numpy._core'] = np.core
            sys.modules['numpy._core.multiarray'] = np.core.multiarray
            sys.modules['numpy._core.numeric'] = np.core.numeric
            sys.modules['numpy._core.umath'] = np.core.umath

    data = np.load(npz_path, allow_pickle=True)
    arr = data[key]
    n = int(arr.shape[0])
    if max_tiles is not None:
        n = min(n, max_tiles)
    out = []
    for i in range(n):
        tile = _ensure_hwc_rgb_uint8(_tile_to_uint8_numpy(arr[i, 0]))
        out.append(tile)
    return out

def random_crops_paired(img_a: np.ndarray, img_b: np.ndarray, crop_size: int, num_crops: int, rng: np.random.Generator):
    h, w, _ = img_a.shape
    out_a, out_b = [], []
    for _ in range(num_crops):
        y = int(rng.integers(0, max(1, h - crop_size + 1)))
        x = int(rng.integers(0, max(1, w - crop_size + 1)))
        
        crop_a = img_a[y:y + crop_size, x:x + crop_size, :]
        crop_b = img_b[y:y + crop_size, x:x + crop_size, :]
        
        if crop_a.shape[0] < crop_size or crop_a.shape[1] < crop_size:
            pad_h = max(0, crop_size - crop_a.shape[0])
            pad_w = max(0, crop_size - crop_a.shape[1])
            crop_a = np.pad(crop_a, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            crop_b = np.pad(crop_b, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        out_a.append(crop_a)
        out_b.append(crop_b)
    return out_a, out_b

def calc_wasserstein_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    dists = []
    for c in range(3):
        h1, _ = np.histogram(img1[:, :, c].flatten(), bins=256, range=(0, 255), density=True)
        h2, _ = np.histogram(img2[:, :, c].flatten(), bins=256, range=(0, 255), density=True)
        dists.append(wasserstein_distance(np.arange(256), np.arange(256), h1, h2))
    return float(np.mean(dists))

def calc_edge_iou(img1: np.ndarray, img2: np.ndarray) -> float:
    edges1 = canny(rgb2gray(img1), sigma=1.0)
    edges2 = canny(rgb2gray(img2), sigma=1.0)
    intersection = np.logical_and(edges1, edges2).sum()
    union = np.logical_or(edges1, edges2).sum()
    return float(intersection / union) if union > 0 else 0.0

def get_nuclear_mask(img_rgb: np.ndarray, nuclei_model_instance, model_type: str, is_ihc: bool) -> np.ndarray:
    if nuclei_model_instance is None:
        return np.zeros(img_rgb.shape[:2], dtype=bool)

    if model_type == "cellpose":
        if is_ihc:
            hed = rgb2hed(img_rgb)
            h_channel = hed[:, :, 0]
            labels, _, _, _ = nuclei_model_instance.eval(h_channel, diameter=None, channels=[0,0], cellprob_threshold=-1.0)
        else:
            labels, _, _, _ = nuclei_model_instance.eval(img_rgb, diameter=None, channels=[3,0])
        return labels > 0

    elif model_type == "omnipose":
        if is_ihc:
            hed = rgb2hed(img_rgb)
            h_channel = hed[:, :, 0]
            masks, _, _, _ = nuclei_model_instance.eval(h_channel, diameter=None, channels=[0,0], omni=True)
        else:
            masks, _, _, _ = nuclei_model_instance.eval(img_rgb, diameter=None, channels=[3,0], omni=True)
        return masks > 0
        
    elif model_type == "stardist":
        if img_rgb.max() > 1.0:
            img_norm = (img_rgb / 255.0).astype(np.float32)
        else:
            img_norm = img_rgb.astype(np.float32)
        labels, _ = nuclei_model_instance.predict_instances(img_norm, verbose=False)
        return labels > 0

    elif model_type == "instanseg":
        res = nuclei_model_instance.eval_small_image(img_rgb, pixel_size=0.5)
        # res is a tuple if return_image_tensor=False? Wait, eval_small_image returns just the tensor of shape (1, 1, H, W)
        if isinstance(res, tuple):
            res = res[0]
        labels = res[0, 0].cpu().numpy()
        return labels > 0

    elif model_type in ["micronet", "hovernet", "hovernet_orig"]:
        try:
            import tempfile
            import os
            import cv2
            import joblib

            with tempfile.TemporaryDirectory() as temp_dir:
                img_path = os.path.join(temp_dir, "temp_tile.png")
                cv2.imwrite(img_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                
                target_dir = os.path.join(temp_dir, "hovernet_out")
                
                res_iter = nuclei_model_instance.predict(
                    [img_path], 
                    mode="tile", 
                    save_dir=target_dir
                )
                res_list = list(res_iter)
                
                if not res_list:
                    return np.zeros(img_rgb.shape[:2], dtype=bool)
                
                _, out_prefix = res_list[0]
                dat_file = f"{out_prefix}.dat"
                
                if os.path.exists(dat_file):
                    out_dict = joblib.load(dat_file)
                    if 'resolution_map' in out_dict:
                        return out_dict['resolution_map'] > 0
                    elif 'instance_map' in out_dict:
                        return out_dict['instance_map'] > 0
                    else:
                        mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
                        if isinstance(out_dict, dict):
                            for inst_id, inst_info in out_dict.items():
                                if isinstance(inst_id, int) and 'contour' in inst_info:
                                    contour = np.array(inst_info['contour'])
                                    cv2.fillPoly(mask, [contour], 1)
                        return mask > 0
                return np.zeros(img_rgb.shape[:2], dtype=bool)
                
        except Exception as e:
            print(f"[ERROR] HoVer-Net prediction failed: {e}")
            return np.zeros(img_rgb.shape[:2], dtype=bool)        
        
    elif model_type == "cppnet":
        try:
            import torch
            from torchvision.transforms.functional import to_tensor
            device = next(nuclei_model_instance.parameters()).device
            if isinstance(img_rgb, np.ndarray):
                if img_rgb.dtype != np.uint8:
                    if img_rgb.max() > 1.0:
                        img_norm = (img_rgb / 255.0).astype(np.float32)
                    else:
                        img_norm = img_rgb.astype(np.float32)
                else:
                    img_norm = img_rgb
                t = to_tensor(img_norm).unsqueeze(0).to(device)
            with torch.no_grad():
                out = nuclei_model_instance(t)
            if "inst" in out:
                mask = (out["inst"].squeeze().cpu().numpy() > 0)
            elif "sem" in out:
                mask = (out["sem"].squeeze().argmax(0).cpu().numpy() > 0)
            else:
                mask = list(out.values())[0].squeeze().cpu().numpy() > 0
            if mask.ndim > 2:
                mask = mask[0]
            return mask
        except Exception as e:
            print(f"[ERROR] CPP-Net prediction failed: {e}")
            return np.zeros(img_rgb.shape[:2], dtype=bool)

    elif model_type == "splinedist":
        try:
            from splinedist.utils import normalize
            img_norm = normalize(img_rgb, 1, 99.8, axis=(0, 1))
            labels, details = nuclei_model_instance.predict_instances(img_norm)
            return labels > 0
        except Exception as e:
            print(f"[ERROR] SplineDist prediction failed: {e}")
            return np.zeros(img_rgb.shape[:2], dtype=bool)
            
    return np.zeros(img_rgb.shape[:2], dtype=bool)

def calc_mask_iou(mask1: np.ndarray, mask2: np.ndarray, dilate_radius: int = 3) -> float:
    if dilate_radius > 0:
        selem = disk(dilate_radius)
        mask1 = binary_dilation(mask1, selem)
        mask2 = binary_dilation(mask2, selem)
        
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(intersection / union) if union > 0 else 0.0

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", type=str, required=True, help="Path alla cartella testIHC")
    ap.add_argument("--real_root", type=str, required=True, help="Path dataset Lion IHC")
    ap.add_argument("--he_root", type=str, required=True, help="Path H&E originali (1-to-1)")
    ap.add_argument("--sub_cancers", type=str, default="ccRCC,chRCC,onco,pRCC")
    ap.add_argument("--npz_key", type=str, default="arr_0")
    
    ap.add_argument("--fid_real_tiles", type=int, default=2100, help="Totale patch reali da estrarre globalmente per calcolare FID")
    ap.add_argument("--only_iou", action="store_true", help="Se impostato, calcola solo Nuclear Mask IoU e salta le altre metriche")
    
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--crop_size", type=int, default=256)
    ap.add_argument("--crops_per_image", type=int, default=4)
    ap.add_argument("--kid_subsets", type=int, default=50)
    ap.add_argument("--kid_subset_size", type=int, default=100)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--out_xlsx", type=str, default="metrics_results_smart.xlsx")
    ap.add_argument("--nuclei_model", type=str, default="cellpose", choices=["cellpose", "hovernet", "hovernet_orig", "instanseg", "omnipose", "cppnet", "splinedist", "stardist", "micronet"], help="Modello da usare per la maschera nucleare")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    sub_cancers = [x.strip() for x in args.sub_cancers.split(",")]

    nuclei_model_instance = None
    if args.nuclei_model == "cellpose":
        if CELLPOSE_AVAILABLE:
            use_gpu = torch.cuda.is_available()
            nuclei_model_instance = models.Cellpose(gpu=use_gpu, model_type='nuclei')
    elif args.nuclei_model == "hovernet":
        if HOVERNET_AVAILABLE:
            nuclei_model_instance = NucleusInstanceSegmentor(pretrained_model='hovernet_fast-pannuke')
    elif args.nuclei_model == "hovernet_orig":
        if HOVERNET_AVAILABLE:
            nuclei_model_instance = NucleusInstanceSegmentor(pretrained_model='hovernet_original-consep')
    elif args.nuclei_model == "instanseg":
        try:
            from instanseg import InstanSeg
            nuclei_model_instance = InstanSeg("brightfield_nuclei")
        except ImportError:
            return
    elif args.nuclei_model == "omnipose":
        try:
            from cellpose_omni import models
            nuclei_model_instance = models.Cellpose(gpu=(device != "cpu"), model_type="cyto2_omni")
        except ImportError:
            return
    elif args.nuclei_model == "micronet":
        if HOVERNET_AVAILABLE:
            nuclei_model_instance = NucleusInstanceSegmentor(pretrained_model="micronet-consep")
            if not hasattr(nuclei_model_instance.ioconfig, "margin"):
                nuclei_model_instance.ioconfig.margin = [0, 0, 0, 0]
    elif args.nuclei_model == "stardist":
        if STARDIST_AVAILABLE:
            nuclei_model_instance = StarDist2D.from_pretrained("2D_versatile_he")
    elif args.nuclei_model == "cppnet":
        if CPPNET_AVAILABLE:
            nuclei_model_instance = CPPNet.from_pretrained("hgsc_v1_efficientnet_b5").to(device)
            nuclei_model_instance.eval()
    elif args.nuclei_model == "splinedist":
        if SPLINEDIST_AVAILABLE:
            try:
                nuclei_model_instance = SplineDist2D.from_pretrained('2D_versatile_he')
            except Exception as e:
                nuclei_model_instance = SplineDist2D(None, name='splinedist_fallback', basedir='.')

    rows = []

    for sub in sub_cancers:
        
        fake_sub_root = Path(args.gen_root) / sub
        real_sub_root = Path(args.real_root) / sub
        he_sub_root = Path(args.he_root) / sub

        fake_bns = list_npz_basenames(str(fake_sub_root))
        if not fake_bns:
            continue
            
        fake_dict = {}
        for bn in fake_bns:
            fake_dict[bn] = load_all_npz_tiles(str(fake_sub_root / bn), args.npz_key)
            
        flat_fake_imgs = [img for imgs in fake_dict.values() for img in imgs]

        real_ihc_imgs = []
        if not args.only_iou:
            real_bns = list_npz_basenames(str(real_sub_root))
            if real_bns:
                tiles_per_file = args.fid_real_tiles // len(real_bns)
                for bn in real_bns:
                    real_ihc_imgs.extend(load_all_npz_tiles(str(real_sub_root / bn), args.npz_key, max_tiles=tiles_per_file))
            else:

        calc_pixel_morpho = (sub in ["onco", "ccRCC"]) and not args.only_iou
        
        calc_nuclei_morpho = sub in ["onco", "ccRCC", "chRCC", "pRCC"]  

        for sd in seeds:
            print(f"[RUN] Sub={sub} | Seed={sd}")
            rng = np.random.default_rng(sd)

            cropfid, kid_mean = None, None

            if real_ihc_imgs:
                fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
                kid_metric = KernelInceptionDistance(subset_size=args.kid_subset_size, subsets=args.kid_subsets, normalize=True).to(device)

                @torch.no_grad()
                def update_metric(imgs, is_real):
                    batch = []
                    for img in imgs:
                        h, w, _ = img.shape
                        for _ in range(args.crops_per_image):
                            y = int(rng.integers(0, max(1, h - args.crop_size + 1)))
                            x = int(rng.integers(0, max(1, w - args.crop_size + 1)))
                            cr = img[y:y+args.crop_size, x:x+args.crop_size, :]
                            batch.append(torch.from_numpy(cr).permute(2, 0, 1).unsqueeze(0))
                            
                            if len(batch) >= 32:
                                t_batch = torch.cat(batch, dim=0).to(device)
                                fid_metric.update(t_batch, real=is_real)
                                kid_metric.update(t_batch, real=is_real)
                                batch = []
                                
                    if batch:
                        t_batch = torch.cat(batch, dim=0).to(device)
                        fid_metric.update(t_batch, real=is_real)
                        kid_metric.update(t_batch, real=is_real)

                rng = np.random.default_rng(sd)
                update_metric(flat_fake_imgs, is_real=False)
                
                rng = np.random.default_rng(sd)
                update_metric(real_ihc_imgs, is_real=True)

                cropfid = float(fid_metric.compute().detach().cpu().item())
                kid_out = kid_metric.compute()
                kid_mean = float(kid_out[0].detach().cpu().item()) if isinstance(kid_out, tuple) else float(kid_out.detach().cpu().item())

            wass_dist, edge_iou, nuc_iou = None, None, None
            
            if calc_pixel_morpho or calc_nuclei_morpho:
                wass_list, edge_list, nuc_list = [], [], []
                rng_morpho = np.random.default_rng(sd)
                
                for fake_bn, fk_imgs in fake_dict.items():
                    he_bn = fake_bn.replace("ihc_", "he_")
                    
                    he_file = he_sub_root / he_bn 
                    
                    if he_file.exists():
                        he_imgs = load_all_npz_tiles(str(he_file), args.npz_key, max_tiles=len(fk_imgs))
                        
                        min_len = min(len(he_imgs), len(fk_imgs))
                        for he_img, fk_img in zip(he_imgs[:min_len], fk_imgs[:min_len]):
                            he_crops, fk_crops = random_crops_paired(he_img, fk_img, args.crop_size, args.crops_per_image, rng_morpho)
                            
                            for hc, fc in zip(he_crops, fk_crops):
                                
                                if calc_pixel_morpho:
                                    wass_list.append(calc_wasserstein_distance(hc, fc))
                                    edge_list.append(calc_edge_iou(hc, fc))
                                
                                if calc_nuclei_morpho:
                                    mask_he = get_nuclear_mask(hc, nuclei_model_instance, args.nuclei_model, is_ihc=False)
                                    mask_fk = get_nuclear_mask(fc, nuclei_model_instance, args.nuclei_model, is_ihc=True)
                                    nuc_list.append(calc_mask_iou(mask_he, mask_fk, dilate_radius=3)) # <--- QUI VIENE USATA LA DILATAZIONE
                    else:
                        print(f"[WARN] File 1-to-1 H&E '{he_bn}' non trovato. Salto morpho per questo file.")

                if wass_list:
                    wass_dist = np.mean(wass_list)
                if edge_list:
                    edge_iou = np.mean(edge_list)
                if nuc_list:
                    nuc_iou = np.mean(nuc_list)

            rows.append({
                "SubCancer": sub,
                "Seed": sd,
                "FID": cropfid,
                "KID": kid_mean,
                "Wasserstein_HE_Fake": wass_dist,
                "Edge_IoU_HE_Fake": edge_iou,
                "Nuclear_Mask_IoU": nuc_iou
            })
            
            if real_ihc_imgs:
                del fid_metric
                del kid_metric
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    df = pd.DataFrame(rows)
    metrics_cols = ["FID", "KID", "Wasserstein_HE_Fake", "Edge_IoU_HE_Fake", "Nuclear_Mask_IoU"]
    
    agg_df = df.groupby(["SubCancer"])[metrics_cols].agg(['mean', 'std']).reset_index()
    
    final_rows = []
    for _, row in agg_df.iterrows():
        formatted_row = {"SubCancer": row["SubCancer"].values[0]}
        for col in metrics_cols:
            mean_val = row[(col, 'mean')]
            std_val = row[(col, 'std')]
            if pd.isna(mean_val) or pd.isna(std_val):
                formatted_row[col] = "N/A"
            else:
                formatted_row[col] = f"{mean_val:.4f} ± {std_val:.4f}"
        final_rows.append(formatted_row)

    global_row = {"SubCancer": "GLOBAL"}
    for col in metrics_cols:
        global_mean = df[col].mean()
        global_std = df[col].std()
        
        if pd.isna(global_mean) or pd.isna(global_std):
            global_row[col] = "N/A"
        else:
            global_row[col] = f"{global_mean:.4f} ± {global_std:.4f}"
            
    final_rows.append(global_row)

    df_final = pd.DataFrame(final_rows)
    df_final.to_excel(args.out_xlsx, index=False)

if __name__ == "__main__":
    main()
import os
import torch
import einops
import timm
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL

# Assumendo che NPZFolderDataset supporti he_dir e ihc_dir come prima
from npz_dataset import NPZFolderDataset

# --- CONFIGURAZIONE ---
HE_DIR = "/home/sg510849/giuSpathis/data/trainingMLP/heFiltered_clean"  # Cartella con i file NPZ filtrati (H&E)
IHC_DIR = "/home/sg510849/giuSpathis/data/trainingMLP/ihc_filtered"  # Cartella con i file NPZ filtrati (IHC)
SAVE_DIR = "/home/sg510849/giuSpathis/data/trainingMLP/precomputed_paired16"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4 # Aumentato per sfruttare la GPU!

@torch.no_grad()
def precompute():
    os.makedirs(os.path.expanduser(SAVE_DIR), exist_ok=True)
    
    # 1. Caricamento VAE (SD3.5)
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-large", subfolder="vae").to(DEVICE, dtype=torch.float16)
    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0.0)
    vae.eval()

    # 2. Caricamento UNI-v2
    print("Loading UNI...")
    timm_kwargs = {
        "img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24,
        "init_values": 1e-5, "embed_dim": 1536, "mlp_ratio": 2.66667 * 2,
        "num_classes": 0, "no_embed_class": True, "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU, "reg_tokens": 8, "dynamic_img_size": True,
    }
    uni_model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs).to(DEVICE, dtype=torch.float16)
    uni_model.eval()

    # Normalizzazione manuale per UNI (come nel tuo script originale)
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE, dtype=torch.float16).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE, dtype=torch.float16).view(1, 3, 1, 1)

    # 3. Dataset ACCOPPIATO
    dataset = NPZFolderDataset(
        folder_path=HE_DIR,
        mask_folder_path=IHC_DIR,
        key="arr_0", 
        max_per_file=300
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Inizio pre-computazione di {len(dataset)} tile...")
    
    # Contatore globale per i nomi dei file
    global_idx = 0

    for batch in tqdm(dataloader):
        he_uint, ihc_uint = batch
        bs = he_uint.shape[0]

        # Convertiamo in float16, normalizziamo [0, 1] e spostiamo sul device
        # Formato originario: B, H, W, C -> Formato target per VAE/UNI: B, C, H, W
        he = (he_uint.to(DEVICE, dtype=torch.float16) / 255.0).permute(0, 3, 1, 2)
        ihc = (ihc_uint.to(DEVICE, dtype=torch.float16) / 255.0).permute(0, 3, 1, 2)

        # ==========================================
        # 1. ESTRAZIONE UNI (H&E e IHC)
        # ==========================================
        # Se vuoi 64 tokens usa d1=8, d2=8. Se vuoi 16 tokens usa d1=4, d2=4
        uni_patches_he = einops.rearrange(he, "b c (d1 h) (d2 w) -> (b d1 d2) c h w", d1=4, d2=4)
        uni_patches_ihc = einops.rearrange(ihc, "b c (d1 h) (d2 w) -> (b d1 d2) c h w", d1=4, d2=4)

        # Resize a 224x224 (richiesto da UNI) e normalizzazione
        uni_in_he = F.interpolate(uni_patches_he, size=(224, 224), mode="bicubic", align_corners=False)
        uni_in_ihc = F.interpolate(uni_patches_ihc, size=(224, 224), mode="bicubic", align_corners=False)
        
        uni_in_he = (uni_in_he - mean) / std
        uni_in_ihc = (uni_in_ihc - mean) / std

        # Calcolo combinato per ottimizzare la GPU
        with torch.autocast(device_type=DEVICE, dtype=torch.float16):
            uni_emb_all = uni_model(torch.cat((uni_in_he, uni_in_ihc), dim=0))
        
        uni_emb_he, uni_emb_ihc = torch.chunk(uni_emb_all, chunks=2, dim=0)
        
        # Reshape a (B, 64, 1536)
        uni_emb_he = uni_emb_he.reshape(bs, 16, -1)
        uni_emb_ihc = uni_emb_ihc.reshape(bs, 16, -1)

        # ==========================================
        # 2. ESTRAZIONE VAE (H&E e IHC)
        # ==========================================
        # Il VAE si aspetta range [-1, 1]
        he_vae_in = (2.0 * he) - 1.0
        ihc_vae_in = (2.0 * ihc) - 1.0

        latents_he = vae.encode(he_vae_in).latent_dist.sample()
        latents_he = (latents_he - vae_shift) * vae_scale

        latents_ihc = vae.encode(ihc_vae_in).latent_dist.sample()
        latents_ihc = (latents_ihc - vae_shift) * vae_scale

        # ==========================================
        # 3. SALVATAGGIO (Un file per coppia/batch)
        # ==========================================
        for b in range(bs):
            save_path = os.path.join(SAVE_DIR, f"paired_tile_{global_idx}.pt")
            
            torch.save({
                "he_uni": uni_emb_he[b].cpu().to(torch.float32),
                "ihc_uni": uni_emb_ihc[b].cpu().to(torch.float32),
                "he_latent": latents_he[b].cpu().to(torch.float32),
                "ihc_latent": latents_ihc[b].cpu().to(torch.float32)
            }, save_path)
            
            global_idx += 1

    print(f"Completato! {global_idx} file salvati in {SAVE_DIR}")

if __name__ == "__main__":
    precompute()
#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from resmlp import SimpleMLP
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from npz_dataset import NPZFolderDataset

def parse_args():
    p = argparse.ArgumentParser(description="MLP Training 16 TOKENS con Densità precomputate e Warmstart")
    p.add_argument("--data_dir", type=str, required=True, help="Cartella embeddings a 16 token")
    p.add_argument("--densities_dir", type=str, required=True, help="Cartella stardist_densities_8x8")
    p.add_argument("--he_dir", type=str, required=True)
    p.add_argument("--ihc_dir", type=str, required=True)
    p.add_argument("--stain", type=str, default="CK7")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--save_every", type=int, default=5)
    p.add_argument("--save_dir", type=str, default="./checkpoints_mlp")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--use_tf32", action="store_true")
    
    p.add_argument("--nuclei_weight", type=float, default=0.25)
    p.add_argument("--nuclei_start_epoch", type=int, default=150)
    p.add_argument("--nuclei_graph_weight", type=float, default=1.0)
    p.add_argument("--nuclei_density_gamma", type=float, default=2.0)
    
    p.add_argument("--warmstart_ckpt", type=str, default=None, help="Percorso del checkpoint .pth da cui ripartire")
    return p.parse_args()

class PrecomputedMLPDataset(Dataset):
    def __init__(self, emb_dir, den_dir, he_dir, ihc_dir, max_per_file=300):
        self.emb_dir = emb_dir
        
        self.index_mapper = NPZFolderDataset(
            folder_path=he_dir, mask_folder_path=ihc_dir,
            max_per_file=max_per_file, verbose=False
        )
        
        self.num_samples = len(self.index_mapper.samples)
        
        self.densities_ram = []
        
        for i in range(self.num_samples):
            img_path, _, internal_idx = self.index_mapper.samples[i]
            
            rel_path = os.path.relpath(img_path, self.index_mapper.folder_path)
            density_file = rel_path.replace(".npz", "_densities.npy")
            density_full_path = os.path.join(den_dir, density_file)
            
            den_arr = np.load(density_full_path, mmap_mode="r")
            den_patch = torch.from_numpy(np.array(den_arr[internal_idx])).float() # (8, 8)
            
            den_patch = den_patch.unsqueeze(0).unsqueeze(0) # (1, 1, 8, 8)
            den_patch_4x4 = F.max_pool2d(den_patch, kernel_size=2).squeeze() # (4, 4)
            
            self.densities_ram.append(den_patch_4x4)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        emb_path = os.path.join(self.emb_dir, f"paired_tile_{idx}.pt")
        data = torch.load(emb_path, weights_only=True)
        he_emb, ihc_emb = data["he_uni"], data["ihc_uni"]
        den_patch = self.densities_ram[idx] # 4x4
        
        return he_emb, ihc_emb, den_patch

def densities_to_patch_weights(densities_4x4, device, threshold=0.15):
    d = densities_4x4.to(device=device, dtype=torch.float32)
    mask = (d > threshold).float()
    weights = d * mask 
    return weights

def neighborhood_structure_loss(pred_map, ref_map, weights):
    p_n = F.normalize(pred_map, dim=-1)
    r_n = F.normalize(ref_map, dim=-1)
    
    ph = (p_n[:,:,:-1,:] * p_n[:,:,1:,:]).sum(-1)
    rh = (r_n[:,:,:-1,:] * r_n[:,:,1:,:]).sum(-1)
    
    pv = (p_n[:,:-1,:,:] * p_n[:,1:,:,:]).sum(-1)
    rv = (r_n[:,:-1,:,:] * r_n[:,1:,:,:]).sum(-1)
    
    wh = weights[:,:,:-1] * weights[:,:,1:]
    wv = weights[:,:-1,:] * weights[:,1:,:]
    
    loss_h = (F.l1_loss(ph, rh, reduction='none') * wh).mean()
    loss_v = (F.l1_loss(pv, rv, reduction='none') * wv).mean()
    
    return loss_h + loss_v

def main():
    args = parse_args()
    device = torch.device(args.device)
    if args.use_tf32: torch.backends.cuda.matmul.allow_tf32 = True

    dataset = PrecomputedMLPDataset(
        emb_dir=args.data_dir, den_dir=args.densities_dir, 
        he_dir=args.he_dir, ihc_dir=args.ihc_dir
    )


    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, 
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    uni_mlp = SimpleMLP(in_channels=1536, time_embed_dim=1024, model_channels=1024, 
                        bottleneck_channels=1024, out_channels=1536, num_res_blocks=6).to(device)
    
    start_epoch = 0
    if args.warmstart_ckpt is not None:
        if os.path.exists(args.warmstart_ckpt):
            print(f"[INFO] Caricamento pesi pre-addestrati da: {args.warmstart_ckpt}")
            state = torch.load(args.warmstart_ckpt, map_location="cpu")
            uni_mlp.load_state_dict(state, strict=False)
            
            try:
                base_name = os.path.basename(args.warmstart_ckpt)
                if "_ep" in base_name:
                    epoch_str = base_name.split("_ep")[-1].split(".")[0]
                    if epoch_str.isdigit():
                        start_epoch = int(epoch_str)
                else:
            except Exception as e:
        else:
    
    opt = torch.optim.AdamW(uni_mlp.parameters(), lr=args.learning_rate)
    autocast_dtype = torch.bfloat16 if args.use_bf16 else torch.float16
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"Epoch [{epoch+1}/{args.num_epochs}]")
        bar = tqdm(dataloader)
        for step, (he_emb, ihc_emb, den_4x4) in enumerate(bar):
            he_emb, ihc_emb = he_emb.to(device, non_blocking=True), ihc_emb.to(device, non_blocking=True)
            bs = he_emb.shape[0]

            h_flat, i_flat = he_emb.view(bs*16, -1), ihc_emb.view(bs*16, -1)
            t = torch.rand((bs*16, 1), device=device)
            xt = t * i_flat + (1-t) * h_flat
            target = i_flat - h_flat

            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                pred = uni_mlp(xt, 999.0 * t.view(-1))
            
            flow_loss = F.mse_loss(pred.float(), target.float())
            total_loss = flow_loss
            nuc_loss_val = 0.0
            
            if epoch >= args.nuclei_start_epoch:
                x1_hat = xt.float() + (1.0 - t.float()) * pred.float()
                
                p_map = x1_hat.view(bs, 4, 4, -1)
                r_map = h_flat.view(bs, 4, 4, -1)
                
                weights = densities_to_patch_weights(den_4x4, device, args.nuclei_density_gamma)
                
                graph = neighborhood_structure_loss(p_map, r_map, weights)
                
                nuc_loss = args.nuclei_graph_weight * graph
                total_loss += args.nuclei_weight * nuc_loss
                nuc_loss_val = nuc_loss.item()

            opt.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(uni_mlp.parameters(), 1.0)
            opt.step()

            bar.set_postfix({"loss": f"{total_loss.item():.4f}", "flow": f"{flow_loss.item():.4f}", "nuc": f"{nuc_loss_val:.4f}"})

        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.save_dir, f"mlp_nuclei_ep{epoch+1}.pth")
            torch.save(uni_mlp.state_dict(), save_path)

    final_path = os.path.join(args.save_dir, f"PRECOMPUTED_CK7_mlp_final.pth")
    torch.save(uni_mlp.state_dict(), final_path)

if __name__ == "__main__":
    main()
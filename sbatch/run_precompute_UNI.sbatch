#!/bin/bash
#SBATCH --job-name=precompute_emb_16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=60:00:00
#SBATCH --output=./precompute_emb_16.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell

cd ~/giuSpathis/PixCell/virtual_staining



python -u precompute_embeddingsUNI_latentsVAE.py

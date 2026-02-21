#!/bin/bash
#SBATCH --job-name=qualEval
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=./qualEval.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell


python generate_figures.py \
  --he_dir /home/sg510849/giuSpathis/data/testing/he \
  --checkpoints \
    /home/sg510849/giuSpathis/data/pixcellGenIhc/CUSTOM_NPZ_CK7_lora_20 \
    /home/sg510849/giuSpathis/data/pixcellGenIhc/CUSTOM_NPZ_CK7_lora_20_noFlow \
    /home/sg510849/giuSpathis/data/pixcellGenIhc/mist_pr \
    /home/sg510849/giuSpathis/data/pixcellGenIhc/mist_er \
  --ckpt_names lora20 noFlow mist_pr mist_er \
  --out_dir /home/sg510849/giuSpathis/qual_figs_v2 \
  --seed 123 \
  --tissue_min 0.60 \
  --center_crop 512 \
  --center_tissue_min 0.60 \
  --rank_checkpoint lora20 \
  --scan_per_file 200 \
  --topk 3


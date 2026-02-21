#!/bin/bash
#SBATCH --job-name=checkImg
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=./output1.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell


python npz_save_30_images.py \
  --npz /home/sg510849/giuSpathis/data/testing/ihc/ihc_24009749.npz \
  --out_dir ~/giuSpathis/sampleLione \
  --n 500 \
#  --bgr

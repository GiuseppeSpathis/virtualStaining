#!/bin/bash
#SBATCH --job-name=checkImg
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=./output1.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell


python ../scriptsPixcell/npz_save_30_images.py \
  --npz /home/sg510849/giuSpathis/resultsMorph/ihc_180140848.npz \
  --out_dir ~/giuSpathis/images/test \
  --n 500 




#!/bin/bash
#SBATCH --job-name=pixcell_pr
#SBATCH --cpus-per-task=10
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=./output.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell


python svs_to_npz.py \
  --path ~/giuSpathis/dataset/CK7_Lyon \
  --mode testing \
  --size 1024

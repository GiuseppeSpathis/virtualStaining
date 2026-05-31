#!/bin/bash
#SBATCH --job-name=svs2npz
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=./svs2npz.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell


python -u /home/sg510849/giuSpathis/scriptsPixcell/svs_to_npz.py \
  --path /home/sg510849/giuSpathis/data/tmp/sheyenneDataset \
  --mode testing \
  --size 1024

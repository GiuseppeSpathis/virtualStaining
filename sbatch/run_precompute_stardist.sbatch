#!/bin/bash
#SBATCH --job-name=stardist_precomp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=./stardist_precomp.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell

cd /home/sg510849/giuSpathis/scriptsPixcell

HE_DIR="/home/sg510849/giuSpathis/data/trainingMLP/heFiltered_clean"
OUT_DIR="/home/sg510849/giuSpathis/data/trainingMLP/stardist_densities_8x8"

echo "Input H&E: $HE_DIR"
echo "Output NPY: $OUT_DIR"

python -u precompute_stardist_densities.py \
    --he_dir "$HE_DIR" \
    --output_dir "$OUT_DIR" \
    --grid_size 8

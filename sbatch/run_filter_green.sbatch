#!/bin/bash
#SBATCH --job-name=filter_green
#SBATCH --output=./filter_green.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00


source ~/.bashrc
conda activate pixcell

cd /home/sg510849/giuSpathis/scriptsPixcell

python filter_green_npz.py \
    --input_dir /home/sg510849/giuSpathis/data/ihc \
    --output_dir /home/sg510849/giuSpathis/data/ihc_filtered \
    --key arr_0 \
    --green_margin 20 \
    --max_green_ratio 0.02 \
    --max_tiles_per_file 0  
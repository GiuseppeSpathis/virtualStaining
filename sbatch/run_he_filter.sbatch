#!/bin/bash
#SBATCH --job-name=filterData
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=./filterData_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell


python -u /home/sg510849/giuSpathis/scriptsPixcell/filter_acellular_he.py

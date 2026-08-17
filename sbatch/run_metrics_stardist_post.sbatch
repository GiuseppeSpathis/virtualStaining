#!/bin/bash
#SBATCH --job-name=fid_morpho_stardist_postNovelties
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=100:00:00
#SBATCH --output=./logs/output_metrics_stardist_postNovelties.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell 

python ../scriptsPixcell/compute_metrics_subcancer.py \
  --gen_root /home/sg510849/giuSpathis/data/trainingMLP/testIHC \
  --real_root /home/sg510849/giuSpathis/data/ihc_filtered \
  --he_root /home/sg510849/giuSpathis/data/trainingMLP/test \
  --sub_cancers ccRCC,chRCC,onco,pRCC \
  --fid_real_tiles 2100 \
  --seeds 0,1,2,3,4 \
  --nuclei_model stardist \
  --out_xlsx /home/sg510849/giuSpathis/data/trainingMLP/results/fid_kid_morpho_results_stardist_postNovelties.xlsx \
  --crop_size 256 \
  --crops_per_image 4 \
  --kid_subsets 50 \
  --kid_subset_size 100 \
  --only_iou

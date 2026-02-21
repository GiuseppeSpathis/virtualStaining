#!/bin/bash
#SBATCH --job-name=fid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00          
#SBATCH --output=./outputfid.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell 

python compute_fid_kid.py \
  --gen_root ~/giuSpathis/data/pixcellGenIhc \
  --real_root ~/giuSpathis/data/testing/ihc \
  --folders mist_er,mist_pr,CUSTOM_NPZ_CK7_lora_20,CUSTOM_NPZ_CK7_lora_20_noFlow \
  --out_xlsx ~/giuSpathis/data/fid_results.xlsx \
  --target_total 2100 \
  --fake_n_files 7 \
  --seeds 0,1,2,3,4 \
  --out_xlsx ~/giuSpathis/data/fid_results_3000.xlsx \
  --kid_subsets 50 \
  --kid_subset_size 100 \
 


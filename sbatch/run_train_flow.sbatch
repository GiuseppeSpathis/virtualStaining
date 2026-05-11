#!/bin/bash
#SBATCH --job-name=fl_ccRCC
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=100:00:00
#SBATCH --output=./flow_ck7_ccRCC.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell

cd ~/giuSpathis/PixCell/virtual_staining


#mkdir -p /home/sg510849/giuSpathis/PixCell/checkpoints_mlp_ck7_pRCC

# Se UNI2-h viene scaricato da HuggingFace e il server lo richiede:

python train_flow_mlp.py \
    --dataset CUSTOM_NPZ_PAIRED \
    --he_dir /home/sg510849/giuSpathis/data/trainingMLP/normalizedHe \
    --ihc_paired_dir /home/sg510849/giuSpathis/data/trainingMLP/normalizedIhc_filtered \
    --stain CK7 \
    --device cuda \
    --train_batch_size 4 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --save_every 5 \
    --save_dir /home/sg510849/giuSpathis/PixCell/checkpoints_mlp_ck7_morph \
    --num_workers 8 \
    --max_per_file 300 \
    --use_bf16 \
    --use_tf32 \
    --warmstart_target mist_her2

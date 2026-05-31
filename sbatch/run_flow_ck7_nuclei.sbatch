#!/bin/bash
#SBATCH --job-name=ck7_nuc_16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=150:00:00
#SBATCH --output=./mlp_nuclei_16.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell

# Percorsi aggiornati per 16 tokens
DATA_DIR="/home/sg510849/giuSpathis/data/trainingMLP/precomputed_paired16"
DEN_DIR="/home/sg510849/giuSpathis/data/trainingMLP/stardist_densities_8x8"
HE_DIR="/home/sg510849/giuSpathis/data/trainingMLP/heFiltered_clean"
IHC_DIR="/home/sg510849/giuSpathis/data/trainingMLP/ihc_filtered"
SAVE_DIR="/home/sg510849/giuSpathis/PixCell/checkpoints_mlp_ck7_nuclei_16tok_4files"

mkdir -p "$SAVE_DIR"
mkdir -p ./logs

cd ~/giuSpathis/PixCell/virtual_staining

echo "Avvio addestramento accelerato a 16 TOKEN con maschere nucleari..."

python train_flow_mlp_nuclei.py \
    --data_dir "$DATA_DIR" \
    --densities_dir "$DEN_DIR" \
    --he_dir "$HE_DIR" \
    --ihc_dir "$IHC_DIR" \
    --stain CK7 \
    --train_batch_size 16 \
    --num_epochs 150 \
    --learning_rate 5e-5 \
    --save_dir "$SAVE_DIR" \
    --num_workers 8 \
    --nuclei_weight 0.70 \
    --nuclei_graph_weight 1.0 \
    --nuclei_start_epoch 100 \
    --use_bf16 \
    --use_tf32

echo "Job completato."
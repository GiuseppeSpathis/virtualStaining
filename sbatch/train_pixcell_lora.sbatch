#!/bin/bash
#SBATCH --job-name=lora_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=./training_lora.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell



cd PixCell


TRAIN_DATA_DIR="/home/sg510849/giuSpathis/data/training"
CHECKPOINT_DIR="./checkpoints_lora"
LOG_DIR="./training_logs"

mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    virtual_staining/train_lora.py \
    --dataset CUSTOM_NPZ \
    --root_dir $TRAIN_DATA_DIR \
    --stain CK7 \
    --train_batch_size 4 \
    --num_epochs 20 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --mixed_precision bf16 \
    --output_dir $LOG_DIR \
    --save_dir $CHECKPOINT_DIR \
    --num_workers 0 \
    --use_tf32 

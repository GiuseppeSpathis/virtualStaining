#!/bin/bash
#SBATCH --job-name=inflpRCC
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=./inference_lora_pRCC.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell

cd ../PixCell

MY_LORA="/home/sg510849/giuSpathis/PixCell/checkpoints_lora_normalized/CUSTOM_NPZ_CK7_lora_9.pth"
#MY_LORA="mist_her2"
#MY_LORA="/home/sg510849/giuSpathis/PixCell/checkpoints_lora_morph_latent/lora_morph_latent_epoch_3.pth"
#MY_LORA="/home/sg510849/giuSpathis/PixCell/checkpoints_lora_morph_latent_005_misther2/lora_morph_latent_epoch_1.pth"
#MY_CONTROLNET="/home/sg510849/giuSpathis/PixCell/checkpoints_controlnet/CUSTOM_NPZ_CK7_controlnet_20.pth"
INPUT_DIR="/home/sg510849/giuSpathis/data/mehdiTesting/pRCC"
OUTPUT_BASE="/home/sg510849/giuSpathis/eval1/pRCC/lora"
DEBUG_FILE="HP24009757-0-0-HE_62441.npz"
#MY_FLOW="/home/sg510849/giuSpathis/PixCell/checkpoints_mlp_ck7_scratch/CUSTOM_NPZ_PAIRED_CK7_mlp_final.pth"
MY_FLOW="mist_her2"

python -u run_pixcell_virtual_stain_npz.py \
  --input_dir $INPUT_DIR \
  --output_base_dir $OUTPUT_BASE \
  --target $MY_LORA \
  --max_images_per_file 20 \
  --flow_target $MY_FLOW \
  --device cuda \
  --num_inference_steps 50 \
  --guidance_scale 1.2 \
  --flow_steps 100 \
  --batch_size 1 \
  --use_bf16 \
  --use_tf32 \
  --bgr \
  ${DEBUG_FILE:+--debug_file "$DEBUG_FILE"}

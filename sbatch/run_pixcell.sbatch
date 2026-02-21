#!/bin/bash
#SBATCH --job-name=gen_noFlow
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00          
#SBATCH --output=./gen_output_noFlow.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell

cd PixCell


INPUT_DIR="/home/sg510849/giuSpathis/data/testing/he"
OUTPUT_BASE="/home/sg510849/giuSpathis/data/pixcellGenIhc"

# flow without lora tuned checkpoint
#python -u run_pixcell_virtual_stain_npz.py \
#  --input_dir $INPUT_DIR \
#  --output_base_dir $OUTPUT_BASE \
#  --target mist_ki67 \
#  --device cuda \
#  --num_inference_steps 100 \
#  --guidance_scale 1.2 \
#  --flow_steps 100 \
#  --batch_size 4 \
#  --use_bf16 \
#  --use_tf32 \
#  --profile \
#  --bgr

# checkpoint LoRA without flow
 MY_CHECKPOINT="/home/sg510849/giuSpathis/PixCell/checkpoints_lora/CUSTOM_NPZ_CK7_lora_20_noFlow.pth"
 python -u run_pixcell_virtual_stain_npz.py \
   --input_dir $INPUT_DIR \
   --output_base_dir $OUTPUT_BASE \
   --target $MY_CHECKPOINT \
   --no_flow \
   --device cuda \
   --num_inference_steps 100 \
   --guidance_scale 1.2 \
   --batch_size 4 \
   --use_bf16 \
   --use_tf32 \
#   --profile \
#   --bgr

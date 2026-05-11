#!/bin/bash
#SBATCH --job-name=pixcell_run
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=100:00:00
#SBATCH --output=./logs/inference_pixcell.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell

cd /home/sg510849/giuSpathis/PixCell

# --- VARIABILI GLOBALI (Condivise da entrambi gli script) ---
MY_LORA="/home/sg510849/giuSpathis/PixCell/checkpoints_lora_normalized/CUSTOM_NPZ_CK7_lora_9.pth"
MY_FLOW="/home/sg510849/giuSpathis/PixCell/checkpoints_mlp_ck7_nuclei_16tok/mlp_nuclei_ep90.pth"
BASE_DATA_DIR="/home/sg510849/giuSpathis/data/trainingMLP"


# ==============================================================================
# L'INTERRUTTORE BOOLEANO
# Imposta a "true"  -> Lancia il loop su tutte le cartelle (Dataset intero)
# Imposta a "false" -> Lancia il test singolo sul file he_180140848.npz
# ==============================================================================
RUN_FULL_DATASET=true


if [ "$RUN_FULL_DATASET" = true ]; then
    echo "=== MODALITA' PRODUZIONE: Elaborazione di tutte le directory ==="

    TUMOR_TYPES=("ccRCC" "chRCC" "onco" "pRCC")

    # 1. ELABORAZIONE TRAINING (heFiltered_clean -> ihc_fake)
    echo "-> Inizio dataset di Training..."
    for TUMOR in "${TUMOR_TYPES[@]}"; do
        INPUT_DIR="${BASE_DATA_DIR}/heFiltered_clean/${TUMOR}"
        OUTPUT_DIR="${BASE_DATA_DIR}/ihc_fake/${TUMOR}"
        
        echo "Processing: $TUMOR -> $OUTPUT_DIR"
        python -u run_pixcell_consensus.py \
          --input_dir "$INPUT_DIR" \
          --output_base_dir "$OUTPUT_DIR" \
          --target "$MY_LORA" \
          --flow_target "$MY_FLOW" \
          --max_images_per_file 300 \
          --device cuda \
          --num_inference_steps 50 \
          --guidance_scale 1.2 \
          --flow_steps 100 \
          --strength 0.70 \
          --seed 2024 \
          --batch_size 1 \
          --progress_threshold 0.73 \
          --stardist_prob_thresh 0.15 \
          --hsv_sat_scale 0.4 \
          --bg_h_preserve 0.2 \
          --use_bf16 \
          --use_tf32 \
          --bgr
    done

    # 2. ELABORAZIONE TEST (test -> testIHC)
    echo "-> Inizio dataset di Test..."
    for TUMOR in "${TUMOR_TYPES[@]}"; do
        INPUT_DIR="${BASE_DATA_DIR}/test/${TUMOR}"
        OUTPUT_DIR="${BASE_DATA_DIR}/testIHC/${TUMOR}"
        
        echo "Processing: $TUMOR -> $OUTPUT_DIR"
        python -u run_pixcell_consensus.py \
          --input_dir "$INPUT_DIR" \
          --output_base_dir "$OUTPUT_DIR" \
          --target "$MY_LORA" \
          --flow_target "$MY_FLOW" \
          --max_images_per_file 300 \
          --device cuda \
          --num_inference_steps 50 \
          --guidance_scale 1.2 \
          --flow_steps 100 \
          --strength 0.70 \
          --seed 2024 \
          --batch_size 1 \
          --progress_threshold 0.73 \
          --stardist_prob_thresh 0.15 \
          --hsv_sat_scale 0.4 \
          --bg_h_preserve 0.2 \
          --use_bf16 \
          --use_tf32 \
          --bgr
    done
    
    echo "=== ELABORAZIONE MASSIVA COMPLETATA ==="

else
    echo "=== MODALITA' DEBUG: Elaborazione file singolo ==="

    INPUT_DIR="/home/sg510849/giuSpathis/data/he/Annotations_chromo"
    OUTPUT_BASE="/home/sg510849/giuSpathis/resultsMorph"
    DEBUG_FILE="he_180140848.npz" 

    python -u run_pixcell_consensus.py \
      --input_dir "$INPUT_DIR" \
      --output_base_dir "$OUTPUT_BASE" \
      --target "$MY_LORA" \
      --flow_target "$MY_FLOW" \
      --max_images_per_file 300 \
      --device cuda \
      --num_inference_steps 50 \
      --guidance_scale 1.2 \
      --flow_steps 100 \
      --strength 0.70 \
      --seed 2024 \
      --batch_size 1 \
      --progress_threshold 0.73 \
      --stardist_prob_thresh 0.15 \
      --hsv_sat_scale 0.4 \
      --bg_h_preserve 0.2 \
      --use_bf16 \
      --use_tf32 \
      --bgr \
      --debug_file "$DEBUG_FILE"
      
    echo "=== DEBUG COMPLETATO ==="

fi
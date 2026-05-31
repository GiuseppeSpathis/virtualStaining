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
export HF_TOKEN=""

# --- VARIABILI GLOBALI ---
MY_LORA="/home/sg510849/giuSpathis/PixCell/checkpoints_lora_normalized/CUSTOM_NPZ_CK7_lora_9.pth"
MY_FLOW="/home/sg510849/giuSpathis/PixCell/checkpoints_mlp_ck7_nuclei_16tok/mlp_nuclei_ep90.pth"
BASE_DATA_DIR="/home/sg510849/giuSpathis/data/trainingMLP"

# Assicurati che questo percorso punti correttamente al tuo script di estrazione
EXTRACT_SCRIPT="/home/sg510849/giuSpathis/scriptsPixcell/npz_save_30_images.py"

# ==============================================================================
# L'INTERRUTTORE BOOLEANO
# Imposta a "true"  -> Lancia il loop su tutte le cartelle (Dataset intero)
# Imposta a "false" -> Lancia il test singolo sul file he_180140848.npz
# ==============================================================================
RUN_FULL_DATASET=true


if [ "$RUN_FULL_DATASET" = true ]; then
    echo "=== MODALITA' PRODUZIONE: Elaborazione di tutte le directory ==="

    TUMOR_TYPES=("ccRCC" "chRCC" "onco" "pRCC")

    # 1. ELABORAZIONE TRAINING (he -> ihc_fake)
    echo "-> Inizio dataset di Training..."
    for TUMOR in "${TUMOR_TYPES[@]}"; do
        INPUT_DIR="${BASE_DATA_DIR}/heFiltered_clean/${TUMOR}"
        OUTPUT_DIR="${BASE_DATA_DIR}/ihc_fake/${TUMOR}"
        
        echo "Processing: $TUMOR -> $OUTPUT_DIR"
        python -u run_pixcell_preprocess.py \
        --input_dir "$INPUT_DIR" \
        --output_base_dir "$OUTPUT_DIR" \
        --target "$MY_LORA" \
        --flow_target "$MY_FLOW" \
        --device cuda \
        --num_inference_steps 50 \
        --guidance_scale 1.2 \
        --flow_steps 100 \
        --strength 0.70 \
        --seed 2024 \
        --consensus_n_seeds 5 \
        --parallel_seeds 5 \
        --dab_vote_min 3 \
        --dab_abs_threshold 0.035 \
        --dab_percentile 70 \
        --consensus_dab_alpha 0.00 \
        --outside_dab_suppress 0.00 \
        --cell_dilate 13 \
        --randomize_consensus_seeds \
        --progress_threshold 0.60 \
        --stardist_prob_thresh 0.15 \
        --hsv_sat_scale 0.4 \
        --bg_h_preserve 0.2 \
        --harmonization_strength 0.40 \
        --batch_size 1 \
        --empty_cuda_cache \
        --use_bf16 \
        --use_tf32 \
        --reference_ihc /home/sg510849/giuSpathis/images/ck7/ihc_24009748/tile_000000_label0.40.png \
        --bgr 

        # --- FASE DI ESTRAZIONE IMMAGINI ---
        echo "-> Estrazione immagini dai file .npz in $OUTPUT_DIR"
        for NPZ_FILE in "$OUTPUT_DIR"/*.npz; do
            # Verifica che il file esista (evita errori se la cartella è vuota)
            [ -e "$NPZ_FILE" ] || continue
            
            # Ottieni il nome del file senza l'estensione (es. ihc_192434_1)
            BASENAME=$(basename "$NPZ_FILE" .npz)
            IMG_OUT_DIR="$OUTPUT_DIR/$BASENAME"
            
            echo "Estraggo $BASENAME in $IMG_OUT_DIR"
            python -u "$EXTRACT_SCRIPT" \
              --npz "$NPZ_FILE" \
              --out_dir "$IMG_OUT_DIR" \
              --n 999999 \
              --bgr
        done
    done

    # 2. ELABORAZIONE TEST (test -> testIHC)
    echo "-> Inizio dataset di Test..."
    for TUMOR in "${TUMOR_TYPES[@]}"; do
        INPUT_DIR="${BASE_DATA_DIR}/test/${TUMOR}"
        OUTPUT_DIR="${BASE_DATA_DIR}/testIHC/${TUMOR}"
        
        echo "Processing: $TUMOR -> $OUTPUT_DIR"
        python -u run_pixcell_preprocess.py \
        --input_dir "$INPUT_DIR" \
        --output_base_dir "$OUTPUT_DIR" \
        --target "$MY_LORA" \
        --flow_target "$MY_FLOW" \
        --device cuda \
        --num_inference_steps 50 \
        --guidance_scale 1.2 \
        --flow_steps 100 \
        --strength 0.70 \
        --seed 2024 \
        --consensus_n_seeds 5 \
        --parallel_seeds 5 \
        --dab_vote_min 3 \
        --dab_abs_threshold 0.035 \
        --dab_percentile 70 \
        --consensus_dab_alpha 0.00 \
        --outside_dab_suppress 0.00 \
        --cell_dilate 13 \
        --randomize_consensus_seeds \
        --progress_threshold 0.60 \
        --stardist_prob_thresh 0.15 \
        --hsv_sat_scale 0.4 \
        --bg_h_preserve 0.2 \
        --harmonization_strength 0.40 \
        --batch_size 1 \
        --empty_cuda_cache \
        --use_bf16 \
        --use_tf32 \
        --reference_ihc /home/sg510849/giuSpathis/images/ck7/ihc_24009748/tile_000000_label0.40.png \
        --bgr 

        # --- FASE DI ESTRAZIONE IMMAGINI ---
        echo "-> Estrazione immagini dai file .npz in $OUTPUT_DIR"
        for NPZ_FILE in "$OUTPUT_DIR"/*.npz; do
            [ -e "$NPZ_FILE" ] || continue
            
            BASENAME=$(basename "$NPZ_FILE" .npz)
            IMG_OUT_DIR="$OUTPUT_DIR/$BASENAME"
            
            echo "Estraggo $BASENAME in $IMG_OUT_DIR"
            python -u "$EXTRACT_SCRIPT" \
              --npz "$NPZ_FILE" \
              --out_dir "$IMG_OUT_DIR" \
              --n 999999 \
              --bgr
        done
    done
    
    echo "=== ELABORAZIONE MASSIVA COMPLETATA ==="

else
    echo "=== MODALITA' DEBUG: Elaborazione file singolo ==="

    INPUT_DIR="/home/sg510849/giuSpathis/data/he/chRCC"
    OUTPUT_BASE="/home/sg510849/giuSpathis/resultsMorph"
    DEBUG_FILE="he_180140848.npz" 

    python -u run_pixcell_preprocess.py \
        --input_dir "$INPUT_DIR" \
        --output_base_dir "$OUTPUT_BASE" \
        --target "$MY_LORA" \
        --flow_target "$MY_FLOW" \
        --debug_file "$DEBUG_FILE" \
        --max_images_per_file 1 \
        --tile_idx 1 \
        --device cuda \
        --num_inference_steps 50 \
        --guidance_scale 1.2 \
        --flow_steps 100 \
        --strength 0.70 \
        --seed 2024 \
        --consensus_n_seeds 5 \
        --parallel_seeds 5 \
        --dab_vote_min 3 \
        --dab_abs_threshold 0.035 \
        --dab_percentile 70 \
        --consensus_dab_alpha 0.00 \
        --outside_dab_suppress 0.00 \
        --cell_dilate 13 \
        --randomize_consensus_seeds \
        --progress_threshold 0.60 \
        --stardist_prob_thresh 0.15 \
        --hsv_sat_scale 0.4 \
        --bg_h_preserve 0.2 \
        --harmonization_strength 0.40 \
        --batch_size 1 \
        --empty_cuda_cache \
        --use_bf16 \
        --use_tf32 \
        --reference_ihc /home/sg510849/giuSpathis/images/ck7/ihc_24009748/tile_000000_label0.40.png \
        --bgr 
      
    # Estrazione per il singolo file in modalità debug
    BASENAME=$(basename "$DEBUG_FILE" .npz)
    BASENAME_IHC="${BASENAME/he_/ihc_}" # Sostituisce he_ con ihc_ per matchare l'output
    
    # Se è stato specificato un tile_idx, il nome cambia, ma per sicurezza cerchiamo l'ultimo npz modificato
    LATEST_NPZ=$(ls -t "$OUTPUT_BASE"/*.npz | head -1)
    
    if [ -n "$LATEST_NPZ" ]; then
        LATEST_BASENAME=$(basename "$LATEST_NPZ" .npz)
        IMG_OUT_DIR="$OUTPUT_BASE/$LATEST_BASENAME"
        
        echo "Estraggo immagini debug da $LATEST_NPZ in $IMG_OUT_DIR"
        python -u "$EXTRACT_SCRIPT" \
            --npz "$LATEST_NPZ" \
            --out_dir "$IMG_OUT_DIR" \
            --n 999999 \
            --bgr
    fi

    echo "=== DEBUG COMPLETATO ==="
fi
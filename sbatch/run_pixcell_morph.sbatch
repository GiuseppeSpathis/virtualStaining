#!/bin/bash
#SBATCH --job-name=pixcell_run_ccRCC
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=100:00:00
#SBATCH --output=./logs/inference_pixcell_debug_ccRCC.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell

cd /home/sg510849/giuSpathis/PixCell
export HF_TOKEN=""

MY_LORA="/home/sg510849/giuSpathis/PixCell/checkpoints_lora_normalized/CUSTOM_NPZ_CK7_lora_9.pth"
MY_FLOW="/home/sg510849/giuSpathis/PixCell/checkpoints_mlp_ck7_nuclei_16tok/mlp_nuclei_ep90.pth"
BASE_DATA_DIR="/home/sg510849/giuSpathis/data/trainingMLP"
EXTRACT_SCRIPT="/home/sg510849/giuSpathis/scriptsPixcell/npz_save_30_images.py"

# ==============================================================================
RUN_FULL_DATASET=false
# ==============================================================================

if [ "$RUN_FULL_DATASET" = true ]; then
    echo "=== MODALITA' PRODUZIONE: Elaborazione di tutte le directory ==="

    #TUMOR_TYPES=("ccRCC" "chRCC" "onco" "pRCC")
    TUMOR_TYPES=("pRCC")

    
    for TUMOR in "${TUMOR_TYPES[@]}"; do
        INPUT_DIR="${BASE_DATA_DIR}/heFiltered_clean/${TUMOR}"
        OUTPUT_DIR="${BASE_DATA_DIR}/ihc_fake_afterNormalization/${TUMOR}"
        mkdir -p "$OUTPUT_DIR"
        
        echo "Processing: $TUMOR -> $OUTPUT_DIR"
        
        for INPUT_NPZ in "$INPUT_DIR"/*.npz; do
            [ -e "$INPUT_NPZ" ] || continue
            
            IN_BASENAME=$(basename "$INPUT_NPZ")
            OUT_BASENAME="${IN_BASENAME/he_/ihc_}"
            OUT_NPZ="$OUTPUT_DIR/$OUT_BASENAME"

            if [ -f "$OUT_NPZ" ]; then
            else
                #python -u run_pixcell_preprocess.py \
                #--input_dir "$INPUT_DIR" \
                #--output_base_dir "$OUTPUT_DIR" \
                #--debug_file "$IN_BASENAME" \
                #--target "$MY_LORA" \
                #--flow_target "$MY_FLOW" \
                #--device cuda \
                #--num_inference_steps 50 \
                #--guidance_scale 1.2 \
                #--flow_steps 100 \
                #--dab_positive_strength 0.80 \
                #--dab_negative_strength 0.55 \
                #--dab_positive_progress=0.60 \
                #--dab_negative_progress=0.40 \
                #--scout_dab_threshold 0.20 \
                #--seed 2024 \
                #--consensus_n_seeds 5 \
                #--parallel_seeds 5 \
                #--dab_vote_min 3 \
                #--dab_abs_threshold 0.05 \
                #--dab_percentile 70 \
                #--consensus_dab_alpha 0.00 \
                #--outside_dab_suppress 0.00 \
                #--cell_dilate 13 \
                #--randomize_consensus_seeds \
                #--stardist_prob_thresh 0.15 \
                #--hsv_sat_scale 0.4 \
                #--bg_h_preserve 0.2 \
                #--harmonization_strength 0.40 \
                #--batch_size 1 \
                #--empty_cuda_cache \
                #--use_bf16 \
                #--use_tf32 \
                #--reference_ihc /home/sg510849/giuSpathis/images/ck7/ihc_24009748/tile_000000_label0.40.png \
                #--bgr 
            fi
        done

        for NPZ_FILE in "$OUTPUT_DIR"/*.npz; do
            [ -e "$NPZ_FILE" ] || continue
            
            BASENAME=$(basename "$NPZ_FILE" .npz)
            IMG_OUT_DIR="$OUTPUT_DIR/$BASENAME"
            
            if [ -d "$IMG_OUT_DIR" ]; then
            else
                python -u "$EXTRACT_SCRIPT" \
                  #--npz "$NPZ_FILE" \
                  #--out_dir "$IMG_OUT_DIR" \
                  #--n 999999 
            fi
        done
    done

    for TUMOR in "${TUMOR_TYPES[@]}"; do
        INPUT_DIR="${BASE_DATA_DIR}/test/${TUMOR}"
        OUTPUT_DIR="${BASE_DATA_DIR}/testIHC_afterNormalization/${TUMOR}"
        mkdir -p "$OUTPUT_DIR"
        
        
        for INPUT_NPZ in "$INPUT_DIR"/*.npz; do
            [ -e "$INPUT_NPZ" ] || continue
            
            IN_BASENAME=$(basename "$INPUT_NPZ")
            OUT_BASENAME="${IN_BASENAME/he_/ihc_}"
            OUT_NPZ="$OUTPUT_DIR/$OUT_BASENAME"

            if [ -f "$OUT_NPZ" ]; then
            else
                python -u run_pixcell_preprocess.py \
                --input_dir "$INPUT_DIR" \
                --output_base_dir "$OUTPUT_DIR" \
                --debug_file "$IN_BASENAME" \
                --target "$MY_LORA" \
                --flow_target "$MY_FLOW" \
                --device cuda \
                --num_inference_steps 50 \
                --guidance_scale 1.2 \
                --flow_steps 100 \
                --dab_positive_strength 0.80 \
                --dab_negative_strength 0.55 \
                --dab_positive_progress=0.60 \
                --dab_negative_progress=0.40 \
                --scout_dab_threshold 0.20 \
                --seed 2024 \
                --consensus_n_seeds 5 \
                --parallel_seeds 5 \
                --dab_vote_min 3 \
                --dab_abs_threshold 0.05 \
                --dab_percentile 70 \
                --consensus_dab_alpha 0.00 \
                --outside_dab_suppress 0.00 \
                --cell_dilate 13 \
                --randomize_consensus_seeds \
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
            fi
        done

        for NPZ_FILE in "$OUTPUT_DIR"/*.npz; do
            [ -e "$NPZ_FILE" ] || continue
            
            BASENAME=$(basename "$NPZ_FILE" .npz)
            IMG_OUT_DIR="$OUTPUT_DIR/$BASENAME"
            
            if [ -d "$IMG_OUT_DIR" ]; then
            else
                python -u "$EXTRACT_SCRIPT" \
                  --npz "$NPZ_FILE" \
                  --out_dir "$IMG_OUT_DIR" \
                  --n 999999 
            fi
        done
    done
    

else

    INPUT_DIR="/home/sg510849/giuSpathis/data/trainingMLP/test/ccRCC"
    OUTPUT_BASE="/home/sg510849/giuSpathis/resultsMorph"
    DEBUG_FILE="he_192434_2.npz" 
    
    OUT_BASENAME="${DEBUG_FILE/he_/ihc_}"
    OUT_NPZ="$OUTPUT_BASE/$OUT_BASENAME"

    if [ -f "$OUT_NPZ" ]; then
    else
        python -u run_pixcell_preprocess.py \
            --input_dir "$INPUT_DIR" \
            --output_base_dir "$OUTPUT_BASE" \
            --target "$MY_LORA" \
            --flow_target "$MY_FLOW" \
            --debug_file "$DEBUG_FILE" \
            --max_images_per_file 1 \
            --tile_idx 2 \
            --device cuda \
            --num_inference_steps 50 \
            --guidance_scale 1.2 \
            --flow_steps 100 \
            --dab_positive_strength 0.80 \
            --dab_negative_strength 0.55 \
            --dab_positive_progress=0.60 \
            --dab_negative_progress=0.40 \
            --show_debug_output \
            --scout_dab_threshold 0.20 \
            --seed 2024 \
            --consensus_n_seeds 5 \
            --parallel_seeds 5 \
            --dab_vote_min 3 \
            --dab_abs_threshold 0.05 \
            --dab_percentile 70 \
            --consensus_dab_alpha 0.00 \
            --outside_dab_suppress 0.00 \
            --cell_dilate 13 \
            --randomize_consensus_seeds \
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
    fi
      
    LATEST_NPZ=$(ls -t "$OUTPUT_BASE"/*.npz 2>/dev/null | head -1)
    
    if [ -n "$LATEST_NPZ" ]; then
        LATEST_BASENAME=$(basename "$LATEST_NPZ" .npz)
        IMG_OUT_DIR="$OUTPUT_BASE/$LATEST_BASENAME"
        
        if [ -d "$IMG_OUT_DIR" ]; then
        else
            python -u "$EXTRACT_SCRIPT" \
                --npz "$LATEST_NPZ" \
                --out_dir "$IMG_OUT_DIR" \
                --n 999999 
        fi
    fi

fi
#!/bin/bash
#SBATCH --job-name=nr_onco
#SBATCH --output=/home/sg510849/giuSpathis/sbatchsPixcell/stain_normalize_onco.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate stain-san
echo "Python: $(which python) | Env: $CONDA_DEFAULT_ENV"

# ─── PARAMETRI ────────────────────────────────────────────────────────────────
INPUT_DIR=/home/sg510849/giuSpathis/data/ihc_filtered/onco
OUTPUT_DIR=/home/sg510849/giuSpathis/data/normalizedIhc_filtered/onco
STAIN_SAN_DIR=/home/sg510849/giuSpathis/stain-san
WORK_DIR=/home/sg510849/giuSpathis/data/stain_san_tmp_onco

N_TILES=1000        # tile per NPZ
N_JOBS=16          # parallelismo CPU (uguale a cpus-per-task)

# ─── RUN ──────────────────────────────────────────────────────────────────────
SCRIPT=/home/sg510849/giuSpathis/scriptsPixcell/stain_normalize.py

python "$SCRIPT" \
    --input_dir      "$INPUT_DIR"      \
    --output_dir     "$OUTPUT_DIR"     \
    --stain_san_dir  "$STAIN_SAN_DIR"  \
    --work_dir       "$WORK_DIR"       \
    --n_jobs         $N_JOBS

EXIT_CODE=$?
echo "Script finished at $(date) with exit code $EXIT_CODE"
exit $EXIT_CODE

#!/bin/bash
#SBATCH --job-name=filterData
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=./filterData_%j.out

# Inizializza conda e attiva l'ambiente
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell

# Stampa un log dell'inizio
echo "Inizio job: $(date)"
echo "Esecuzione filtraggio dataset con StarDist..."

# Lancia lo script Python (assicurati che il path sia corretto)
python -u /home/sg510849/giuSpathis/scriptsPixcell/filter_acellular_he.py

# Stampa un log di fine
echo "Fine job: $(date)"
#!/bin/bash
#SBATCH --job-name=precompute_emb_16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=60:00:00
#SBATCH --output=./precompute_emb_16.out

# Attivazione ambiente
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pixcell

# Spostati nella cartella di lavoro
cd ~/giuSpathis/PixCell/virtual_staining

# Esporta il token (fondamentale per scaricare/verificare UNI2-h)

echo "Inizio pre-estrazione degli embedding UNI e latenti VAE..."

# Avvia lo script python (assicurati che il nome del file coincida)
python -u precompute_embeddingsUNI_latentsVAE.py

echo "Estrazione completata con successo!"
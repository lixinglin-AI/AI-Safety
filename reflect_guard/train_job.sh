#!/bin/bash
#SBATCH --job-name=reflect_train
#SBATCH --partition=scavenge_gpu
#SBATCH --account=pi_lg689
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.out

module load miniconda
conda activate reflect_guard

cd "$HOME/AI-Safety-Abuse"

# Retrain with existing data, verify adapter from disk
python reflect_guard/main.py --skip-synthesis --hf-token "$HF_TOKEN"

#!/bin/bash
#SBATCH --job-name=reflect_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/vast/palmer/home.mccleary/ll2276/AI-Safety-Abuse/logs/train_%j.out

module load miniconda
conda activate reflect_guard

cd /vast/palmer/home.mccleary/ll2276/AI-Safety-Abuse

# Retrain with existing data, verify adapter from disk
python reflect_guard/main.py --skip-synthesis --hf-token "$HF_TOKEN"

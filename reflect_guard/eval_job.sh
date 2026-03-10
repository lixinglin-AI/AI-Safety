#!/bin/bash
#SBATCH --job-name=reflect_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=/vast/palmer/home.mccleary/ll2276/AI-Safety-Abuse/logs/eval_%j.out

module load miniconda
conda activate reflect_guard

cd /vast/palmer/home.mccleary/ll2276/AI-Safety-Abuse

# Run evaluation (tokens passed via env var to avoid interactive prompt)
python reflect_guard/evaluate_reflect_guard.py --dataset both --hf-token "$HF_TOKEN"

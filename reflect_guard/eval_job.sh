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

# Step 1: Baseline JailbreakBench (WildGuard baseline already exists as wildguard_results_baseline.json)
python reflect_guard/evaluate_baseline_hpc.py --dataset jailbreakbench --hf-token "$HF_TOKEN"

# Step 2: Reflect-Guard on both benchmarks
python reflect_guard/evaluate_reflect_guard.py --dataset both --hf-token "$HF_TOKEN"

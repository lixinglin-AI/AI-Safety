#!/bin/bash
#SBATCH --job-name=reflect_eval_jbb
#SBATCH --partition=scavenge_gpu
#SBATCH --account=pi_lg689
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_jbb_%j.out

module load miniconda
conda activate reflect_guard

cd "$HOME/AI-Safety-Abuse"

# Baseline JailbreakBench
python reflect_guard/evaluate_baseline_hpc.py --dataset jailbreakbench --hf-token "$HF_TOKEN"

# Reflect-Guard JailbreakBench
python reflect_guard/evaluate_reflect_guard.py --dataset jailbreakbench --hf-token "$HF_TOKEN"

# Re-generate comparison report with all results
python analyze_results.py --compare

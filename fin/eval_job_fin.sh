#!/bin/bash
#SBATCH --job-name=rg_fin_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/vast/palmer/home.mccleary/ll2276/AI-Safety-Abuse/logs/fin_eval_%j.out

module load miniconda
conda activate reflect_guard

cd /vast/palmer/home.mccleary/ll2276/AI-Safety-Abuse

# Step 1: Build benchmark (if not already built)
if [ ! -f fin/fin_benchmark.jsonl ]; then
    echo "Building FinSafetyBench benchmark..."
    python fin/build_benchmark.py --hf-token "$HF_TOKEN" --openai-key "$OPENAI_API_KEY"
fi

# Step 2: Baseline evaluation
echo "Running baseline evaluation..."
python fin/evaluate_fin.py --model baseline --hf-token "$HF_TOKEN"

# Step 3: ReflectGuard evaluation
echo "Running ReflectGuard evaluation..."
python fin/evaluate_fin.py --model reflect --hf-token "$HF_TOKEN"

# Step 4: Analysis and LaTeX tables
echo "Generating analysis and LaTeX tables..."
python fin/analyze_fin_results.py

echo "Done. Results in results/fin_results_*.json and results/fin_analysis.txt"

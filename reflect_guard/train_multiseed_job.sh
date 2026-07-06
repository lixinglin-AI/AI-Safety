#!/bin/bash
#SBATCH --job-name=reflect_train_multiseed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=/vast/palmer/home.mccleary/ll2276/AI-Safety-Abuse/logs/train_multiseed_%j.out

module load miniconda
conda activate reflect_guard

cd /vast/palmer/home.mccleary/ll2276/AI-Safety-Abuse

# Train + evaluate 2 additional seeds for variance estimation.
# The original default run (implicit seed=42 paths) is left untouched and is
# reused as the third data point — no need to retrain it.
for SEED in 123 2024; do
    echo "=== Seed $SEED: training ==="
    python reflect_guard/main.py --skip-synthesis --train-only --seed "$SEED" --hf-token "$HF_TOKEN"

    echo "=== Seed $SEED: evaluating on WildGuardTest + JailbreakBench ==="
    python reflect_guard/evaluate_reflect_guard.py --dataset both --seed "$SEED" --hf-token "$HF_TOKEN"
done

echo "Done. Results:"
echo "  results/wildguard_results_reflect.json        (original, seed=42)"
echo "  results/wildguard_results_reflect_seed123.json"
echo "  results/wildguard_results_reflect_seed2024.json"
echo "  results/jailbreakbench_results_reflect.json        (original, seed=42)"
echo "  results/jailbreakbench_results_reflect_seed123.json"
echo "  results/jailbreakbench_results_reflect_seed2024.json"
echo
echo "Next: python reflect_guard/compute_seed_variance.py"

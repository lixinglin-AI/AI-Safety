#!/bin/bash
#SBATCH --job-name=reflect_train_multiseed
#SBATCH --partition=gpu_devel
#SBATCH --account=pi_lg689
#SBATCH --qos=normal
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_multiseed_%j.out

# gpu_devel instead of scavenge_gpu: this account has no grant on the regular
# `gpu` partition, and scavenge_gpu is preemptible — jobs were observed getting
# PREEMPTED after 15-55 minutes in practice, before a single seed's training
# could finish. gpu_devel is open to all accounts (AllowAccounts=ALL) and is NOT
# preemptible (PreemptMode=OFF), at the cost of a hard 6-hour wall-time cap
# (QOSMaxWallDurationPerJobLimit on the `normal` QOS there — 8h was rejected
# outright, 4h was accepted). If this run doesn't finish in 6h, the
# resume_from_checkpoint logic in train.py/train_ablations.py (save_steps=50)
# means resubmitting the same command continues from the last checkpoint rather
# than restarting — just `sbatch` this script again.

module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate reflect_guard

cd "$HOME/AI-Safety-Abuse"

# sbatch only inherits env vars exported in the submitting shell at submit time —
# easy to forget between sessions. Fall back to .env if HF_TOKEN wasn't already
# exported, so a forgotten `export HF_TOKEN=...` doesn't waste a queue slot.
if [ -z "${HF_TOKEN:-}" ] && [ -f .env ]; then
    set -a
    source .env
    set +a
fi

set -euo pipefail

# Train + evaluate 2 additional seeds for Conditions B (SFT-only) and D
# (Reflect-Guard) for statistical-significance testing. The original seed=42 run
# is reused as-is (unsuffixed paths) rather than retrained — see
# reflect_guard/train_ablations.py and main.py --seed for how paths are chosen.
#
# Covers WildGuardTest + JailbreakBench + finance for both conditions, so
# reflect_guard/compute_seed_variance.py has everything it needs afterward,
# including for the finance benchmark specifically (329 examples — the
# reviewer-flagged case most sensitive to single-run noise).
for SEED in 123 2024; do
    echo "============================================"
    echo "SEED $SEED: train Condition B (SFT-only)"
    echo "============================================"
    python reflect_guard/train_ablations.py --ablation b --seed "$SEED" --hf-token "$HF_TOKEN"

    echo ""
    echo "============================================"
    echo "SEED $SEED: train Condition D (Reflect-Guard)"
    echo "============================================"
    python reflect_guard/main.py --skip-synthesis --train-only --skip-smoke-test --seed "$SEED" --hf-token "$HF_TOKEN"

    echo ""
    echo "============================================"
    echo "SEED $SEED: evaluate Condition B on all 3 benchmarks"
    echo "============================================"
    python reflect_guard/evaluate_ablations.py --condition b --dataset all --seed "$SEED" --hf-token "$HF_TOKEN"

    echo ""
    echo "============================================"
    echo "SEED $SEED: evaluate Condition D on all 3 benchmarks"
    echo "============================================"
    python reflect_guard/evaluate_ablations.py --condition d --dataset all --seed "$SEED" --hf-token "$HF_TOKEN"
done

echo ""
echo "============================================"
echo "Aggregating seed variance (42 + 123 + 2024)"
echo "============================================"
python reflect_guard/compute_seed_variance.py --seeds 42 123 2024 --dataset all

echo ""
echo "ALL DONE. See results/seed_variance_report.txt"

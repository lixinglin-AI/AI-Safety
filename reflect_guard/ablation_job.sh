#!/bin/bash
#SBATCH --job-name=reflect-ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out

module load miniconda

# This allows the shell to recognize the 'conda' command
source $(conda info --base)/etc/profile.d/conda.sh

conda activate reflect_guard

cd "$HOME/AI-Safety-Abuse"

set -euo pipefail

echo "============================================"
echo "PHASE 1: Synthesize ablation training data"
echo "============================================"

# Ablation B (labels only) - no API call needed
python reflect_guard/synthesize_ablations.py --ablation b
echo "Ablation B data generated."

# Ablation C (blind reflections) - needs OpenAI API
if [ -n "${OPENAI_API_KEY:-}" ]; then
    python reflect_guard/synthesize_ablations.py --ablation c
    echo "Ablation C data generated."
else
    echo "WARNING: OPENAI_API_KEY not set. Skipping ablation C synthesis."
    echo "Run manually: OPENAI_API_KEY=... python reflect_guard/synthesize_ablations.py --ablation c"
fi

echo ""
echo "============================================"
echo "PHASE 2: Train ablation B adapter"
echo "============================================"
python reflect_guard/train_ablations.py --ablation b --hf-token "$HF_TOKEN"

echo ""
echo "============================================"
echo "PHASE 3: Train ablation C adapter"
echo "============================================"
if [ -f "reflect_guard/ablation_c_train.jsonl" ]; then
    python reflect_guard/train_ablations.py --ablation c --hf-token "$HF_TOKEN"
else
    echo "Skipping ablation C training (no data file)."
fi

echo ""
echo "============================================"
echo "PHASE 4: Evaluate all conditions"
echo "============================================"

# Evaluate each condition separately for robustness
for cond in 0 a b c d; do
    echo ""
    echo "--- Evaluating condition: $cond ---"
    if [ "$cond" = "c" ] && [ ! -d "reflect_guard/ablation_c_lora" ]; then
        echo "Skipping condition C (no adapter)."
        continue
    fi
    python reflect_guard/evaluate_ablations.py --condition "$cond" --dataset both \
        --hf-token "$HF_TOKEN" || {
        echo "WARNING: Condition $cond failed. Continuing..."
    }
done

echo ""
echo "ALL DONE."

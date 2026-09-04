#!/bin/bash
#SBATCH --job-name=reflect-baselines
#SBATCH --partition=gpu_devel
#SBATCH --account=pi_lg689
#SBATCH --qos=normal
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --output=%x_%j.out

# Runs the ShieldGemma competitive baseline on WildGuardTest + JailbreakBench +
# finance. WildGuard is deliberately NOT run here — see baselines/wildguard_job.sh,
# which uses a separate conda env, because the `wildguard` PyPI package pulls in
# vllm, whose pinned dependency versions can conflict with the bitsandbytes/peft/
# transformers stack this repo otherwise uses. PromptGuard is not run here either —
# it's already been run (v1) and its results are in results/baselines/.
#
# gpu_devel (not scavenge_gpu): this account has no grant on the regular `gpu`
# partition, and scavenge_gpu is preemptible — jobs were observed getting
# PREEMPTED after 15-55 minutes in practice. gpu_devel is open to all accounts
# and NOT preemptible, at the cost of a hard 6h wall-time cap (QOSMaxWallDurationPerJobLimit
# on the `normal` QOS there), hence --time=05:30:00 here for buffer.
#
# GPU type is pinned to a40: this cluster's shared GPU pools mix in Blackwell
# (rtx_pro_6000_blackwell/b200), which the cu118 torch build pinned in
# reflect_guard/setup_bouchet.sh does NOT support (max supported compute
# capability sm_90; Blackwell is sm_120) — an unpinned `--gres=gpu:1` request can
# land you on an incompatible node.
#
# Prereqs (one-time):
#   conda activate reflect_guard
#   pip install -r requirements.txt   # (after adding `openai`, see repo root)

module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate reflect_guard

cd "$HOME/AI-Safety-Abuse"

# sbatch only inherits env vars exported in the submitting shell at submit time.
if [ -z "${HF_TOKEN:-}" ] && [ -f .env ]; then
    set -a
    source .env
    set +a
fi

set -euo pipefail

echo "============================================"
echo "ShieldGemma-9b"
echo "============================================"
python baselines/evaluate_shieldgemma.py --dataset all --model-size 9b --hf-token "$HF_TOKEN" || {
    echo "WARNING: ShieldGemma-9b failed (likely OOM). Retrying with 2b..."
    python baselines/evaluate_shieldgemma.py --dataset all --model-size 2b --hf-token "$HF_TOKEN"
}

echo ""
echo "ALL DONE. Results in results/baselines/."

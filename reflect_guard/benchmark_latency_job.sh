#!/bin/bash
#SBATCH --job-name=reflect-latency
#SBATCH --partition=gpu_devel
#SBATCH --account=pi_lg689
#SBATCH --qos=normal
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/latency_%j.out

# gpu_devel (not scavenge_gpu): this account has no grant on the regular `gpu`
# partition; gpu_devel is open to all accounts and NOT preemptible
# (PreemptMode=OFF), capped at 6h wall time by its QOS — this job is inference
# only (no training) and should finish well under that.
# GPU pinned to a40: this cluster's scavenge/devel pools mix in Blackwell nodes
# (rtx_pro_6000_blackwell/b200) that the pinned cu118 torch build doesn't support
# (max sm_90; Blackwell is sm_120) — an unpinned `--gres=gpu:1` can land you there.

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

# All three conditions run back-to-back in the same job/GPU/driver state so the
# comparison is apples-to-apples (no cross-job hardware/driver variance).
python reflect_guard/benchmark_latency.py --condition all --n-prompts 200 --batch-size 8 --hf-token "$HF_TOKEN"

echo ""
echo "DONE. Results in results/latency/latency_benchmark_0_B_D.json"

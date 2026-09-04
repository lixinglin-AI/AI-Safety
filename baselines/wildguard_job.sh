#!/bin/bash
#SBATCH --job-name=reflect-wildguard-baseline
#SBATCH --partition=gpu_devel
#SBATCH --account=pi_lg689
#SBATCH --qos=normal
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=05:30:00
#SBATCH --output=%x_%j.out

# Runs the WildGuard competitive baseline in its own conda env (the `wildguard`
# PyPI package depends on vllm, which pins torch/transformers versions that can
# conflict with the bitsandbytes/peft stack the rest of this repo uses — keeping it
# isolated avoids breaking the main `reflect_guard` env).
#
# gpu_devel (not scavenge_gpu): not preemptible, open to all accounts, capped at
# 6h wall time (hence --time=05:30:00 for buffer) — see baselines_job.sh for the
# same reasoning in more detail.
#
# GPU type is pinned to a40: this cluster's shared GPU pools mix in Blackwell
# nodes that vllm's own CUDA build here may or may not support depending on what
# got installed — pin to a known-good architecture rather than risk it.
#
# One-time setup (python 3.11+ required — jailbreakbench-adjacent tooling that
# vllm's dependency chain pulls in needs typing.NotRequired, py3.10 doesn't have
# it; also install libstdcxx-ng first or vllm's import chain hits a
# CXXABI_1.3.15-not-found error against this cluster's system libstdc++):
#   conda create -n reflect_guard_wildguard python=3.11 -y
#   conda activate reflect_guard_wildguard
#   conda install -c conda-forge -y "libstdcxx-ng>=13"
#   pip install wildguard datasets huggingface_hub tqdm
#   # then, since a conda env doesn't auto-set LD_LIBRARY_PATH on activation:
#   mkdir -p $CONDA_PREFIX/etc/conda/activate.d
#   echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Do NOT additionally `pip install jailbreakbench` — it depends on litellm, whose
# transformers/protobuf pins conflict with vllm's, and it isn't needed anyway:
# JailbreakBench prompts are fetched directly over HTTP by
# utils/jailbreakbench_loader.py.

module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate reflect_guard_wildguard

cd "$HOME/AI-Safety-Abuse"

if [ -z "${HF_TOKEN:-}" ] && [ -f .env ]; then
    set -a
    source .env
    set +a
fi

set -euo pipefail

# flashinfer's top-k/top-p sampling kernel JIT-compiles on first use, which needs
# a full CUDA toolkit (nvcc) on PATH — this compute node only has the CUDA runtime
# bundled with the pinned torch/vllm wheels, so JIT compilation fails with
# "Could not find nvcc". Falling back to vLLM's native PyTorch sampler avoids this
# entirely and has no effect on classification results (only affects how the next
# token is sampled during generation).
export VLLM_USE_FLASHINFER_SAMPLER=0

python baselines/evaluate_wildguard.py --dataset all --hf-token "$HF_TOKEN"

echo ""
echo "DONE. Results in results/baselines/wildguard_on_*.json"

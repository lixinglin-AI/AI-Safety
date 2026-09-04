#!/bin/bash
# Runs the OpenAI Moderation API baseline. No GPU needed — this calls the OpenAI
# API, so it can run on an HPC login node, locally, or anywhere with network
# access and an OPENAI_API_KEY. Not a SLURM job.
#
# Free endpoint — no per-call token billing, unlike run_prompted_cot.sh's
# gpt-4o-mini calls — so there's no cost reason to price out a subset first, but
# --limit/--smoke-test are still supported for fast iteration/debugging.
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   export HF_TOKEN=...        # only needed for the WildGuardTest portion
#   bash baselines/run_openai_moderation.sh

set -euo pipefail
cd "$(dirname "$0")/.."

python baselines/evaluate_openai_moderation.py --dataset all --concurrency 8 "$@"

echo ""
echo "DONE. Results in results/baselines/openai_moderation_on_*.json"

#!/bin/bash
# Runs the prompted reasoning-based-classifier baseline (GPT-4o-mini via OpenAI API).
# No GPU needed — this calls the OpenAI API, so it can run on an HPC login node,
# locally, or anywhere with network access and an OPENAI_API_KEY. Not a SLURM job.
#
# Usage:
#   export OPENAI_API_KEY=sk-...
#   export HF_TOKEN=...        # only needed for the WildGuardTest portion
#   bash baselines/run_prompted_cot.sh
#
# Cost control: pass --limit N (e.g. --limit 200) to price out a subset before
# running the full ~2,300-call sweep (1,699 WildGuardTest + 282 JailbreakBench +
# 329 finance). Estimate at gpt-4o-mini's per-token price before running full-scale.

set -euo pipefail
cd "$(dirname "$0")/.."

python baselines/evaluate_prompted_cot.py --dataset all --concurrency 8 "$@"

echo ""
echo "DONE. Results in results/baselines/promptedcot_on_*.json"

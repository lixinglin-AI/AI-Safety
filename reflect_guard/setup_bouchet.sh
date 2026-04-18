#!/bin/bash
# Run this once on Bouchet login node to set up the reflect_guard environment.
# Usage: bash reflect_guard/setup_bouchet.sh

set -euo pipefail

module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create env (skip if already exists)
if conda env list | grep -q "^reflect_guard "; then
    echo "Environment reflect_guard already exists, skipping create."
else
    conda create -y -n reflect_guard python=3.10
fi

conda activate reflect_guard

pip install --upgrade pip

pip install \
    torch==2.2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install \
    transformers==4.40.0 \
    peft==0.10.0 \
    bitsandbytes==0.43.1 \
    datasets==2.19.0 \
    accelerate==0.29.3 \
    trl==0.8.6 \
    openai \
    scikit-learn \
    sentencepiece \
    protobuf

echo ""
echo "Done. To activate: module load miniconda && conda activate reflect_guard"
echo "Test with: python -c \"import torch; print(torch.cuda.is_available())\""

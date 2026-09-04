#!/bin/bash
#SBATCH --job-name=ablation-report
#SBATCH --partition=day
#SBATCH --account=pi_lg689
#SBATCH --qos=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/ablation_report_%j.out

module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate reflect_guard

cd "$HOME/AI-Safety-Abuse"

echo "Generating ablation report..."
python analyze_results.py --ablation
echo "Done."

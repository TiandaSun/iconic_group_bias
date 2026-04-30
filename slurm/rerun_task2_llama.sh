#!/usr/bin/env bash
#SBATCH --job-name=t2_llama
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --account=cs-ontrel-2021
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:1

# =============================================================================
# Task 2 Re-run: LLaMA-3.2-Vision-11B (max_tokens=4096)
# =============================================================================

set -e

cd ${PROJECT_ROOT:-$PWD}

module load Miniconda3
source activate vlm_eval

export MKL_THREADING_LAYER=GNU
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in your environment before submitting (needed for gated LLaMA-3.2-Vision).}"

MODEL="llama-3.2-vision-11b"
DATA="data/metadata.csv"
OUTPUT="results/raw"

echo "=============================================="
echo "Task 2 Re-run: ${MODEL} (max_tokens=4096)"
echo "Start Time: $(date)"
echo "=============================================="

echo ""
echo "=== Task 2: Description (Chinese) ==="
python scripts/run_inference.py \
    --task 2 \
    --model ${MODEL} \
    --language zh \
    --data ${DATA} \
    --output ${OUTPUT} \
    --num-samples 500 \
    --checkpoint-interval 50

echo ""
echo "=== Task 2: Description (English) ==="
python scripts/run_inference.py \
    --task 2 \
    --model ${MODEL} \
    --language en \
    --data ${DATA} \
    --output ${OUTPUT} \
    --num-samples 500 \
    --checkpoint-interval 50

echo ""
echo "=============================================="
echo "Pipeline Complete: ${MODEL}"
echo "End Time: $(date)"
echo "=============================================="
echo "Results saved to: ${OUTPUT}"

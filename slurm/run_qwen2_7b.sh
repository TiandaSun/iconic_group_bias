#!/usr/bin/env bash
#SBATCH --job-name=vlm_qwen2_7b
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
# VLM Evaluation Pipeline: Qwen2-VL-7B
# Runs both Task 1 (Classification) and Task 2 (Description) for both languages
# =============================================================================

set -e  # Exit on error

cd ${PROJECT_ROOT:-$PWD}

module load Miniconda3
source activate vlm_eval

# Fix Intel MKL library conflict
export MKL_THREADING_LAYER=GNU

MODEL="qwen2-vl-7b"
DATA="data/metadata.csv"
OUTPUT="results/raw"

echo "=============================================="
echo "VLM Pipeline: ${MODEL}"
echo "Start Time: $(date)"
echo "=============================================="

# -----------------------------------------------------------------------------
# Task 1: Classification
# -----------------------------------------------------------------------------
echo ""
echo "=== Task 1: Classification (Chinese) ==="
python scripts/run_inference.py \
    --task 1 \
    --model ${MODEL} \
    --language zh \
    --data ${DATA} \
    --output ${OUTPUT} \
    --checkpoint-interval 100

echo ""
echo "=== Task 1: Classification (English) ==="
python scripts/run_inference.py \
    --task 1 \
    --model ${MODEL} \
    --language en \
    --data ${DATA} \
    --output ${OUTPUT} \
    --checkpoint-interval 100

# -----------------------------------------------------------------------------
# Task 2: Description Generation
# -----------------------------------------------------------------------------
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

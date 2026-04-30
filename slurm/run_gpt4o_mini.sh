#!/usr/bin/env bash
#SBATCH --job-name=vlm_gpt4o_mini
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --account=cs-ontrel-2021
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err
#SBATCH --partition=nodes

# =============================================================================
# VLM Evaluation Pipeline: GPT-4o-mini (API)
# Runs both Task 1 (Classification) and Task 2 (Description) for both languages
# =============================================================================

set -e  # Exit on error

cd ${PROJECT_ROOT:-$PWD}

module load Miniconda3
source activate vlm_eval

# Fix Intel MKL library conflict
export MKL_THREADING_LAYER=GNU

# API Key
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY in your environment before submitting.}"

MODEL="gpt-4o-mini"
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
    --num-samples 500 \
    --checkpoint-interval 100 \
    --resume

echo ""
echo "=== Task 1: Classification (English) ==="
python scripts/run_inference.py \
    --task 1 \
    --model ${MODEL} \
    --language en \
    --data ${DATA} \
    --output ${OUTPUT} \
    --num-samples 500 \
    --checkpoint-interval 100 \
    --resume

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
    --checkpoint-interval 50 \
    --resume

echo ""
echo "=== Task 2: Description (English) ==="
python scripts/run_inference.py \
    --task 2 \
    --model ${MODEL} \
    --language en \
    --data ${DATA} \
    --output ${OUTPUT} \
    --num-samples 500 \
    --checkpoint-interval 50 \
    --resume

echo ""
echo "=============================================="
echo "Pipeline Complete: ${MODEL}"
echo "End Time: $(date)"
echo "=============================================="
echo "Results saved to: ${OUTPUT}"

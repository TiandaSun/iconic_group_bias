#!/usr/bin/env bash
#SBATCH --job-name=vlm_test_api
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

cd ${PROJECT_ROOT:-$PWD}

module load Miniconda3
source activate vlm_eval

# Fix Intel MKL library conflict
export MKL_THREADING_LAYER=GNU

echo "=============================================="
echo "API Models Test - $(date)"
echo "Testing: GPT-4o-mini, Gemini-2.5-Flash, Claude-Haiku-4.5"
echo "=============================================="

# Test with small sample (10 images total, using --limit)
LIMIT=10

echo ""
echo "=== GPT-4o-mini (Classification, Chinese) ==="
python scripts/run_inference.py \
    --task 1 \
    --model gpt-4o-mini \
    --language zh \
    --data data/metadata.csv \
    --output results/test_api/ \
    --limit ${LIMIT} \
    --checkpoint-interval 5

echo ""
echo "=== Gemini-2.5-Flash (Classification, Chinese) ==="
python scripts/run_inference.py \
    --task 1 \
    --model gemini-2.5-flash \
    --language zh \
    --data data/metadata.csv \
    --output results/test_api/ \
    --limit ${LIMIT} \
    --checkpoint-interval 5

echo ""
echo "=== Claude-Haiku-4.5 (Classification, Chinese) ==="
python scripts/run_inference.py \
    --task 1 \
    --model claude-haiku-4.5 \
    --language zh \
    --data data/metadata.csv \
    --output results/test_api/ \
    --limit ${LIMIT} \
    --checkpoint-interval 5

echo ""
echo "=============================================="
echo "API Models Test Complete - $(date)"
echo "=============================================="

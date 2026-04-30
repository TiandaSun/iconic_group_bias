#!/usr/bin/env bash
#SBATCH --job-name=vlm_test
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
echo "VLM Pipeline Test - $(date)"
echo "=============================================="

# Use small test dataset (25 images, 5 per ethnic group)
TEST_DATA="data/metadata_test.csv"

# Alternatively, use --limit flag with full dataset
# FULL_DATA="data/metadata.csv"

echo ""
echo "=== Test 1: Qwen2.5-VL-7B Classification (Chinese) ==="
python scripts/run_inference.py \
    --task 1 \
    --model qwen2.5-vl-7b \
    --language zh \
    --data ${TEST_DATA} \
    --output results/test/ \
    --checkpoint-interval 10

echo ""
echo "=== Test 2: Qwen2.5-VL-7B Classification (English) ==="
python scripts/run_inference.py \
    --task 1 \
    --model qwen2.5-vl-7b \
    --language en \
    --data ${TEST_DATA} \
    --output results/test/ \
    --checkpoint-interval 10

echo ""
echo "=== Test 3: Qwen2.5-VL-7B Description (5 samples) ==="
python scripts/run_inference.py \
    --task 2 \
    --model qwen2.5-vl-7b \
    --language zh \
    --data ${TEST_DATA} \
    --output results/test/ \
    --num-samples 5 \
    --checkpoint-interval 5

echo ""
echo "=============================================="
echo "Pipeline Test Complete - $(date)"
echo "=============================================="
echo ""
echo "Check results in: results/test/"

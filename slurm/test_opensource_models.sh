#!/usr/bin/env bash
#SBATCH --job-name=vlm_test_opensource
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

module purge
module load Miniconda3
source activate vlm_eval

# Fix Intel MKL/JIT library conflicts
export MKL_THREADING_LAYER=GNU
export LD_PRELOAD=""
export ENABLE_INTEL_JIT_CONTROL=0
export INTEL_DISABLE_JIT=1

# Unset any Intel-related preloads that might conflict
unset INTEL_LICENSE_FILE 2>/dev/null || true

echo "=============================================="
echo "Open-source Models Test - $(date)"
echo "Testing: Qwen2.5-VL-7B, Qwen2-VL-7B, LLaMA-3.2-Vision-11B"
echo "=============================================="

# Test with small sample using test metadata (25 images)
TEST_DATA="data/metadata_test.csv"

echo ""
echo "=== Qwen2.5-VL-7B (Classification, Chinese) ==="
python scripts/run_inference.py \
    --task 1 \
    --model qwen2.5-vl-7b \
    --language zh \
    --data ${TEST_DATA} \
    --output results/test_opensource/ \
    --checkpoint-interval 10

echo ""
echo "=== Qwen2-VL-7B (Classification, Chinese) ==="
python scripts/run_inference.py \
    --task 1 \
    --model qwen2-vl-7b \
    --language zh \
    --data ${TEST_DATA} \
    --output results/test_opensource/ \
    --checkpoint-interval 10

echo ""
echo "=== LLaMA-3.2-Vision-11B (Classification, Chinese) ==="
python scripts/run_inference.py \
    --task 1 \
    --model llama-3.2-vision-11b \
    --language zh \
    --data ${TEST_DATA} \
    --output results/test_opensource/ \
    --checkpoint-interval 10

echo ""
echo "=============================================="
echo "Open-source Models Test Complete - $(date)"
echo "=============================================="

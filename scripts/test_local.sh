#!/usr/bin/env bash
# Local test script (without SLURM)
# Use this for quick debugging before submitting HPC jobs
#
# Usage:
#   bash scripts/test_local.sh              # Test with 5 images
#   bash scripts/test_local.sh 10           # Test with 10 images
#   bash scripts/test_local.sh 10 gpt-4o-mini  # Test specific model

set -e

cd "$(dirname "$0")/.."

LIMIT=${1:-5}
MODEL=${2:-"gpt-4o-mini"}  # Default to API model (no GPU needed)

echo "=============================================="
echo "Local Pipeline Test"
echo "Model: ${MODEL}"
echo "Limit: ${LIMIT} images"
echo "=============================================="

# Create test data if not exists
if [ ! -f "data/metadata_test.csv" ]; then
    echo "Creating test metadata..."
    python3 scripts/create_test_data.py 5
fi

echo ""
echo "=== Testing Classification (Chinese prompt) ==="
python3 scripts/run_inference.py \
    --task 1 \
    --model ${MODEL} \
    --language zh \
    --data data/metadata.csv \
    --output results/local_test/ \
    --limit ${LIMIT} \
    --checkpoint-interval 5 \
    --log-level DEBUG

echo ""
echo "=== Testing Classification (English prompt) ==="
python3 scripts/run_inference.py \
    --task 1 \
    --model ${MODEL} \
    --language en \
    --data data/metadata.csv \
    --output results/local_test/ \
    --limit ${LIMIT} \
    --checkpoint-interval 5 \
    --log-level DEBUG

echo ""
echo "=============================================="
echo "Local Test Complete!"
echo "Results saved to: results/local_test/"
echo "=============================================="

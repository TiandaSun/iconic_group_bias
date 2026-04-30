#!/usr/bin/env bash
#SBATCH --job-name=t2_gpt4o
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
# Task 2 Re-run: GPT-4o-mini (max_tokens=4096)
# =============================================================================

set -e

cd ${PROJECT_ROOT:-$PWD}

module load Miniconda3
source activate vlm_eval

export MKL_THREADING_LAYER=GNU
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY in your environment before submitting.}"

MODEL="gpt-4o-mini"
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

#!/usr/bin/env bash
#SBATCH --job-name=c3_shuffle_os
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
#SBATCH --time=10:00:00
#SBATCH --account=cs-ontrel-2021
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:1

# =============================================================================
# C3 sanity check: MCQ-option-order shuffled classification
# Runs Task 1 with per-image shuffled A-E options on the THREE open-source
# VLMs, 100 stratified images per model per language, both zh and en.
#
# Addresses peer-review concern that the fixed A:Miao ... E:Tibetan order
# could be driving the observed Miao-mode collapse (LLaMA-en=91% Miao).
# A small position_A_fraction under shuffling (~0.20) would confirm the
# finding reflects content, not position.
#
# API models (GPT-4o-mini, Claude-Haiku-4.5) run separately tomorrow.
# =============================================================================

set -e

cd ${PROJECT_ROOT:-$PWD}

module load Miniconda3
source activate vlm_eval

export MKL_THREADING_LAYER=GNU
export PYTHONPATH=${PROJECT_ROOT:-$PWD}:${PYTHONPATH:-}
# HuggingFace token needed for gated LLaMA-3.2-Vision repo
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in your environment before submitting (needed for gated LLaMA-3.2-Vision).}"

OUTPUT_DIR="results/raw/shuffled_mcq"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "C3 shuffle sanity: open-source VLMs"
echo "Start: $(date)"
echo "=============================================="

for MODEL in qwen2.5-vl-7b qwen2-vl-7b llama-3.2-vision-11b; do
  echo ""
  echo "----------------------------------------------"
  echo "Model: ${MODEL}"
  echo "----------------------------------------------"

  python scripts/run_classification_shuffled.py \
      --model "${MODEL}" \
      --language both \
      --data data/metadata.csv \
      --output "${OUTPUT_DIR}" \
      --num-samples 100 \
      --seed 42
done

echo ""
echo "=============================================="
echo "Running analysis (compare shuffle vs fixed)"
echo "=============================================="
python scripts/analyze_position_bias.py \
    --shuffle-dir "${OUTPUT_DIR}" \
    --out-csv results/metrics/shuffle_sanity_summary.csv \
    --out-md results/metrics/shuffle_sanity_summary.md

echo ""
echo "=============================================="
echo "End: $(date)"
echo "=============================================="

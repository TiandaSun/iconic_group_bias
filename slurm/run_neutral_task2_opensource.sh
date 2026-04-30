#!/usr/bin/env bash
#SBATCH --job-name=c4_neutral_os
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
# C4 sanity check: neutral-prompt Task 2 description
# Addresses peer-review concern that the original Task 2 prompt's
# Miao-coded seed vocabulary ("embroidery, batik, brocade, silver ornaments")
# could drive the Dong->Miao in-text confusion rate (42% zh, 37% en).
#
# Runs the 3 open-source VLMs with the neutral prompt variant on 100
# stratified images per language. API models run separately tomorrow.
# =============================================================================

set -e

cd ${PROJECT_ROOT:-$PWD}

module load Miniconda3
source activate vlm_eval

export MKL_THREADING_LAYER=GNU
export PYTHONPATH=${PROJECT_ROOT:-$PWD}:${PYTHONPATH:-}
# HuggingFace token needed for gated LLaMA-3.2-Vision repo
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in your environment before submitting (needed for gated LLaMA-3.2-Vision).}"

OUTPUT_DIR="results/raw/neutral_prompt"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "C4 neutral-prompt Task 2: open-source VLMs"
echo "Start: $(date)"
echo "=============================================="

for MODEL in qwen2.5-vl-7b qwen2-vl-7b llama-3.2-vision-11b; do
  for LANG in zh en; do
    echo ""
    echo "----------------------------------------------"
    echo "Model: ${MODEL}  Lang: ${LANG}"
    echo "----------------------------------------------"

    python -m src.inference.task2_description \
        --model "${MODEL}" \
        --language "${LANG}" \
        --data data/metadata.csv \
        --output "${OUTPUT_DIR}" \
        --num-samples 100 \
        --seed 42 \
        --prompt-variant neutral
  done
done

echo ""
echo "=============================================="
echo "Running analysis (default vs neutral prompt)"
echo "=============================================="
python scripts/analyze_prompt_leak.py \
    --default-glob "results/raw/task2_*_results.json" \
    --neutral-glob "${OUTPUT_DIR}/task2_neutral_*_results.json" \
    --out-csv results/metrics/prompt_leak_comparison.csv \
    --out-md results/metrics/prompt_leak_comparison.md

echo ""
echo "=============================================="
echo "End: $(date)"
echo "=============================================="

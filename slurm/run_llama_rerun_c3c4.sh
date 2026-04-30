#!/usr/bin/env bash
#SBATCH --job-name=llama_rerun_c3c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
#SBATCH --time=4:00:00
#SBATCH --account=cs-ontrel-2021
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:1

# =============================================================================
# LLaMA-3.2-Vision only: re-run C3 MCQ shuffle + C4 neutral-prompt Task 2.
# Yesterday's run failed due to missing HF_TOKEN (gated repo), so the
# entries for LLaMA in the shuffle_sanity_summary and prompt_leak tables
# are currently spurious zeros — this job repopulates them.
# =============================================================================

set -e

cd ${PROJECT_ROOT:-$PWD}

module load Miniconda3
source activate vlm_eval

export MKL_THREADING_LAYER=GNU
export PYTHONPATH=${PROJECT_ROOT:-$PWD}:${PYTHONPATH:-}
export HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN in your environment before submitting (needed for gated LLaMA-3.2-Vision).}"

SHUFFLE_DIR="results/raw/shuffled_mcq"
NEUTRAL_DIR="results/raw/neutral_prompt"

# --- C3: shuffled-MCQ classification on LLaMA -------------------------------
echo "=============================================="
echo "C3 LLaMA rerun (shuffled MCQ)"
echo "Start: $(date)"
echo "=============================================="

# Remove the spurious zero-accuracy JSONs from yesterday so analysis
# picks up the new files cleanly.
rm -f "${SHUFFLE_DIR}"/task1_shuffled_llama-3.2-vision-11b_*_results.json

python scripts/run_classification_shuffled.py \
    --model llama-3.2-vision-11b \
    --language both \
    --data data/metadata.csv \
    --output "${SHUFFLE_DIR}" \
    --num-samples 100 \
    --seed 42

# --- C4: neutral-prompt Task 2 on LLaMA -------------------------------------
echo ""
echo "=============================================="
echo "C4 LLaMA rerun (neutral-prompt Task 2)"
echo "=============================================="

# Yesterday's LLaMA task2 files were produced from an empty model load
# (the pipeline still wrote summary JSONs with zero successful results).
# Remove them so this run's results replace them cleanly.
rm -f "${NEUTRAL_DIR}"/task2_neutral_llama-3.2-vision-11b_*_results.json
rm -f "${NEUTRAL_DIR}"/checkpoints/*llama*

for LANG in zh en; do
  python -m src.inference.task2_description \
      --model llama-3.2-vision-11b \
      --language "${LANG}" \
      --data data/metadata.csv \
      --output "${NEUTRAL_DIR}" \
      --num-samples 100 \
      --seed 42 \
      --prompt-variant neutral
done

# --- Re-run both analysis scripts with the updated LLaMA outputs ------------
echo ""
echo "=============================================="
echo "Re-running analyses with updated LLaMA data"
echo "=============================================="

python scripts/analyze_position_bias.py \
    --shuffle-dir "${SHUFFLE_DIR}" \
    --out-csv results/metrics/shuffle_sanity_summary.csv \
    --out-md results/metrics/shuffle_sanity_summary.md

python scripts/analyze_prompt_leak.py \
    --default-glob "results/raw/task2_*_results.json" \
    --neutral-glob "${NEUTRAL_DIR}/task2_neutral_*_results.json" \
    --out-csv results/metrics/prompt_leak_comparison.csv \
    --out-md results/metrics/prompt_leak_comparison.md

echo ""
echo "=============================================="
echo "End: $(date)"
echo "=============================================="

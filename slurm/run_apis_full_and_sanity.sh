#!/usr/bin/env bash
#SBATCH --job-name=apis_c2c6_c3_c4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --account=cs-ontrel-2021
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err
#SBATCH --partition=nodes

# =============================================================================
# API job: addresses peer-review concerns C2 + C6 + C3(APIs) + C4(APIs)
#
# Sequence (defensive: each phase continues on prior failure):
#
# 1. C6/C2 PRIMARY: Claude-Haiku-4.5 Task 1 at full n=3000 × 2 langs
#    - Eliminates the n=500 vs n=3000 sample asymmetry for Claude
#    - Eliminates the claude-3.5-sonnet → claude-haiku-4.5 relabeling
#      integrity issue (new data is genuinely haiku-4.5)
#    - Estimated cost: ~$4-5 (budget $10.51)
#
# 2. C6 PRIMARY: GPT-4o-mini Task 1 at full n=3000 × 2 langs
#    - Eliminates the sample asymmetry for GPT
#    - Estimated cost: ~$3 (budget $4.19) — TIGHT, runs after Claude
#
# 3. C3 SANITY: shuffled-MCQ on both APIs, 100 images × 2 langs
#    - Estimated cost: <$0.30 total
#
# 4. C4 SANITY: neutral-prompt Task 2 on both APIs, 100 images × 2 langs
#    - Estimated cost: ~$2 total
#
# Total estimated spend: ~$5 Claude + ~$5 OpenAI (tight on OpenAI but
# margin-of-error check at each phase below).
# =============================================================================

set +e  # DO NOT exit on failure — we want later phases to still run

cd ${PROJECT_ROOT:-$PWD}

module load Miniconda3
source activate vlm_eval

export MKL_THREADING_LAYER=GNU
export PYTHONPATH=${PROJECT_ROOT:-$PWD}:${PYTHONPATH:-}
export OPENAI_API_KEY="${OPENAI_API_KEY:?Set OPENAI_API_KEY in your environment before submitting.}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY in your environment (e.g., via a private .env file or your secret manager) before submitting.}"

DATA="data/metadata.csv"
FULL_OUT="results/raw/full_n3000"
SHUFFLE_OUT="results/raw/shuffled_mcq"
NEUTRAL_OUT="results/raw/neutral_prompt"

mkdir -p "${FULL_OUT}" "${SHUFFLE_OUT}" "${NEUTRAL_OUT}"

# -----------------------------------------------------------------------------
# PHASE 1: Claude-Haiku-4.5 Task 1 — full n=3000 × 2 langs  (C2 + C6)
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "PHASE 1  [$(date)]  Claude-Haiku-4.5 Task 1 full n=3000"
echo "Est. cost: ~\$4-5. Budget check: \$10.51 Anthropic."
echo "=============================================="
for LANG in zh en; do
  python scripts/run_inference.py \
      --task 1 \
      --model claude-haiku-4.5 \
      --language ${LANG} \
      --data ${DATA} \
      --output "${FULL_OUT}" \
      --checkpoint-interval 200
  echo "  Phase 1 / ${LANG} exit: $?"
done

# -----------------------------------------------------------------------------
# PHASE 2: GPT-4o-mini Task 1 — full n=3000 × 2 langs  (C6)
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "PHASE 2  [$(date)]  GPT-4o-mini Task 1 full n=3000"
echo "Est. cost: ~\$3. Budget check: \$4.19 OpenAI."
echo "=============================================="
for LANG in zh en; do
  python scripts/run_inference.py \
      --task 1 \
      --model gpt-4o-mini \
      --language ${LANG} \
      --data ${DATA} \
      --output "${FULL_OUT}" \
      --checkpoint-interval 200
  echo "  Phase 2 / ${LANG} exit: $?"
done

# -----------------------------------------------------------------------------
# PHASE 3: C3 shuffled-MCQ sanity on both APIs (100 stratified images × 2 langs)
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "PHASE 3  [$(date)]  C3 shuffled-MCQ on APIs"
echo "Est. cost: <\$0.30 total."
echo "=============================================="
for MODEL in claude-haiku-4.5 gpt-4o-mini; do
  python scripts/run_classification_shuffled.py \
      --model ${MODEL} \
      --language both \
      --data ${DATA} \
      --output "${SHUFFLE_OUT}" \
      --num-samples 100 \
      --seed 42
  echo "  Phase 3 / ${MODEL} exit: $?"
done

# -----------------------------------------------------------------------------
# PHASE 4: C4 neutral-prompt Task 2 sanity on both APIs (100 × 2 langs each)
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "PHASE 4  [$(date)]  C4 neutral-prompt Task 2 on APIs"
echo "Est. cost: ~\$2 total."
echo "=============================================="
for MODEL in claude-haiku-4.5 gpt-4o-mini; do
  for LANG in zh en; do
    python -m src.inference.task2_description \
        --model ${MODEL} \
        --language ${LANG} \
        --data ${DATA} \
        --output "${NEUTRAL_OUT}" \
        --num-samples 100 \
        --seed 42 \
        --prompt-variant neutral
    echo "  Phase 4 / ${MODEL} / ${LANG} exit: $?"
  done
done

# -----------------------------------------------------------------------------
# PHASE 5: Re-run downstream analyses
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "PHASE 5  [$(date)]  Re-run analyses with new API data"
echo "=============================================="

# Shuffle summary (now includes API models)
python scripts/analyze_position_bias.py \
    --shuffle-dir "${SHUFFLE_OUT}" \
    --out-csv results/metrics/shuffle_sanity_summary.csv \
    --out-md results/metrics/shuffle_sanity_summary.md

# Prompt-leak summary (now includes API models)
python scripts/analyze_prompt_leak.py \
    --default-glob "results/raw/task2_*_results.json" \
    --neutral-glob "${NEUTRAL_OUT}/task2_neutral_*_results.json" \
    --out-csv results/metrics/prompt_leak_comparison.csv \
    --out-md results/metrics/prompt_leak_comparison.md

echo ""
echo "=============================================="
echo "END  [$(date)]"
echo "=============================================="
echo "Check outputs:"
echo "  - ${FULL_OUT}/task1_claude-haiku-4.5_*_results.json"
echo "  - ${FULL_OUT}/task1_gpt-4o-mini_*_results.json"
echo "  - ${SHUFFLE_OUT}/task1_shuffled_{claude-haiku-4.5,gpt-4o-mini}_*_results.json"
echo "  - ${NEUTRAL_OUT}/task2_neutral_{claude-haiku-4.5,gpt-4o-mini}_*_results.json"
echo "  - results/metrics/shuffle_sanity_summary.md"
echo "  - results/metrics/prompt_leak_comparison.md"

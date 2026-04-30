#!/usr/bin/env bash
#SBATCH --job-name=regen_metrics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --account=cs-ontrel-2021
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err
#SBATCH --partition=nodes

# =============================================================================
# Regenerate all analysis outputs from the new Task 1 n=3000 data on all 5
# models plus the existing Task 2 data. CPU-only, ~10-15 min wall-clock.
#
# Pipeline:
#   1. compute_classification_metrics.py    -> task1_per_cell / _bias_metrics
#   2. run_statistical_tests.py             -> pairwise McNemar + OBI variants
#   3. salience_prevalence_test.py          -> Wikipedia proxy scatter (refetch)
#   4. salience_prevalence_robust.py        -> exact permutation + Holm + LOO
#   5. compute_description_metrics.py       -> Task 2 per-description metrics
#   6. analyze_text_confusion.py            -> in-text confusion + geo figs
#   7. analyze_rater_calibration.py         -> split-annotation HE report
#   8. audit_dataset.py                     -> dedup audit + DATASHEET draft
#   9. generate_figures.py                  -> all paper figures
# =============================================================================

set +e

cd ${PROJECT_ROOT:-$PWD}

module load Miniconda3
source activate vlm_eval

export MKL_THREADING_LAYER=GNU
export PYTHONPATH=${PROJECT_ROOT:-$PWD}:${PYTHONPATH:-}

# Install one-off helpers we need for C7
pip install --quiet imagehash Pillow 2>/dev/null || true

echo "=============================================="
echo "Regenerating ALL metrics from new n=3000 data"
echo "Start: $(date)"
echo "=============================================="

run_step() {
    echo ""
    echo "--- [$(date +%H:%M:%S)] $1 ---"
    shift
    "$@"
    echo "    exit: $?"
}

# Phase 1 — Task 1 metrics, bias metrics, per-cell accuracy
run_step "compute_classification_metrics" \
    python scripts/compute_classification_metrics.py

# Phase 2 — Pairwise McNemar + OBI variants + LES + bootstrap CIs
run_step "run_statistical_tests" \
    python scripts/run_statistical_tests.py

# Phase 3 — Wikipedia prevalence + figure (this re-fetches Wikipedia; slow)
run_step "salience_prevalence_test" \
    python scripts/salience_prevalence_test.py

# Phase 4 — Exact permutation Spearman + Holm + log-linear + jackknife
run_step "salience_prevalence_robust" \
    python scripts/salience_prevalence_robust.py

# Phase 5 — Task 2 description metrics (length, TTR, CTC, correct-ethnic CTC)
run_step "compute_description_metrics" \
    python scripts/compute_description_metrics.py

# Phase 6 — In-text cross-group confusion + geo-mislocation
run_step "analyze_text_confusion" \
    python scripts/analyze_text_confusion.py

# Phase 7 — Split-annotation human-eval calibration (split-half proxy)
run_step "analyze_rater_calibration" \
    python scripts/analyze_rater_calibration.py

# Phase 8 — Dataset audit (perceptual hash dedup + datasheet draft)
run_step "audit_dataset" \
    python scripts/audit_dataset.py

# Phase 9 — Re-generate all paper figures
run_step "generate_figures" \
    python scripts/generate_figures.py

# Phase 10 — Refresh the shuffle + prompt-leak summaries with canonical names
run_step "analyze_position_bias" \
    python scripts/analyze_position_bias.py

run_step "analyze_prompt_leak" \
    python scripts/analyze_prompt_leak.py

echo ""
echo "=============================================="
echo "End: $(date)"
echo "=============================================="
echo ""
echo "Key outputs to inspect:"
echo "  results/metrics/task1_per_cell.csv"
echo "  results/metrics/task1_bias_metrics.json"
echo "  results/metrics/task1_significance.json     (OBI variants + logit + h)"
echo "  results/metrics/salience_prevalence.json    (ρ per proxy)"
echo "  results/metrics/salience_prevalence_robust.csv (exact p + Holm + LOO)"
echo "  results/metrics/shuffle_sanity_summary.md"
echo "  results/metrics/prompt_leak_comparison.md"
echo "  results/human_eval/rater_calibration_report.md"
echo "  data/DATASHEET.md"
echo "  results/figures/*.pdf / *.png"

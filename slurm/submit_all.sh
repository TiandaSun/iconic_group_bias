#!/usr/bin/env bash
# =============================================================================
# Submit all VLM evaluation jobs
# Each job runs both Task 1 (Classification) and Task 2 (Description)
# for both languages (Chinese and English)
#
# Usage: bash slurm/submit_all.sh [--opensource-only | --api-only]
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
OPENSOURCE_ONLY=false
API_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --opensource-only)
            OPENSOURCE_ONLY=true
            shift
            ;;
        --api-only)
            API_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash slurm/submit_all.sh [--opensource-only | --api-only]"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "VLM Evaluation Pipeline - Job Submission"
echo "=============================================="
echo ""

# Submit open-source models (require GPU)
if [ "$API_ONLY" = false ]; then
    echo "Submitting open-source model jobs (GPU required)..."
    echo "  - Qwen2.5-VL-7B"
    sbatch "${SCRIPT_DIR}/run_qwen25_7b.sh"
    echo "  - Qwen2-VL-7B"
    sbatch "${SCRIPT_DIR}/run_qwen2_7b.sh"
    echo "  - LLaMA-3.2-Vision-11B"
    sbatch "${SCRIPT_DIR}/run_llama.sh"
    echo ""
fi

# Submit API models (no GPU required)
if [ "$OPENSOURCE_ONLY" = false ]; then
    echo "Submitting API model jobs (no GPU required)..."
    echo "  - GPT-4o-mini"
    sbatch "${SCRIPT_DIR}/run_gpt4o_mini.sh"
    echo "  - Claude-Haiku-4.5"
    sbatch "${SCRIPT_DIR}/run_claude.sh"
    echo ""
fi

echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "Each job runs:"
echo "  - Task 1: Classification (zh + en) on 3,000 images"
echo "  - Task 2: Description (zh + en) on 500 sampled images"
echo ""
echo "Monitor jobs:    squeue -u \$USER"
echo "Cancel all:      scancel -u \$USER"
echo "View logs:       tail -f vlm_*-*.log"
echo ""
echo "Results will be saved to: results/raw/"

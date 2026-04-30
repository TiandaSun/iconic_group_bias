#!/usr/bin/env bash
# =============================================================================
# Submit Task 2 re-runs for all 5 models (max_tokens=4096)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=============================================="
echo "Task 2 Re-run Submission (max_tokens=4096)"
echo "=============================================="
echo ""

echo "Submitting open-source models (GPU)..."
echo "  - Qwen2.5-VL-7B"
sbatch "${SCRIPT_DIR}/rerun_task2_qwen25_7b.sh"
echo "  - Qwen2-VL-7B"
sbatch "${SCRIPT_DIR}/rerun_task2_qwen2_7b.sh"
echo "  - LLaMA-3.2-Vision-11B"
sbatch "${SCRIPT_DIR}/rerun_task2_llama.sh"
echo ""

echo "Submitting API models (no GPU)..."
echo "  - GPT-4o-mini"
sbatch "${SCRIPT_DIR}/rerun_task2_gpt4o_mini.sh"
echo "  - Claude-Haiku-4.5"
sbatch "${SCRIPT_DIR}/rerun_task2_claude.sh"
echo ""

echo "=============================================="
echo "All 5 Task 2 re-run jobs submitted!"
echo "=============================================="
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f t2_*-*.log"

#!/usr/bin/env bash
# Generate human evaluation materials

cd ${PROJECT_ROOT:-$PWD}

module load Miniconda3
source activate vlm_eval

python -m src.human_eval.generate_sheets \
    --results-dir results/raw \
    --output-dir results/human_eval \
    --seed 42

echo ""
echo "Generated files:"
ls -la results/human_eval/

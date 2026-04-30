#!/usr/bin/env bash
#SBATCH --job-name=gen_human_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --account=cs-ontrel-2021
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.err
#SBATCH --partition=nodes

cd ${PROJECT_ROOT:-$PWD}

module load Miniconda3
source activate vlm_eval

echo "Generating human evaluation materials..."
echo "Results dir: results/raw"
echo "Output dir: results/human_eval"
echo ""

python -m src.human_eval.generate_sheets \
    --results-dir results/raw \
    --output-dir results/human_eval \
    --seed 42

echo ""
echo "=========================================="
echo "Generated files:"
echo "=========================================="
ls -la results/human_eval/

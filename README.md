# Iconic-Group Bias in Vision-Language Models

Code, predictions, and analysis pipelines for the paper **"Iconic-Group Bias in
Vision-Language Models: How Salience Gradients Render Chinese Ethnic Minority
Cultures Invisible"** (under review, *AI & Society*).

This repository accompanies the paper and contains the full evaluation
pipeline, the per-image model predictions, the aggregated metrics, the
Wikipedia-prevalence salience snapshot, the human-evaluation ratings, and the
analysis scripts that reproduce every figure and table.

The 3,000 source images that constitute the benchmark itself are **not
redistributed** here. The images were collected from publicly available web
sources without a free, prior, and informed consent protocol with the depicted
minority communities; redistributing them at scale would compound the
representational asymmetries the paper documents. Image-level provenance
(per-image source URLs, SHA-256 digests, perceptual-hash signatures, EXIF
metadata, resolution distributions) is recorded in `data/DATASHEET.md` and
`data/dataset_audit.csv`, sufficient for researchers wishing to reconstruct
the corpus or substitute a community-curated alternative.

## Headline finding

Across 30,000 classification inferences on five chat-interface VLMs (Qwen2.5-VL,
Qwen2-VL, LLaMA-3.2-Vision, GPT-4o-mini, Claude-Haiku-4.5) and five Chinese
ethnic minorities (Miao, Dong, Yi, Li, Tibetan), per-group accuracy is
rank-predicted **exactly** (Spearman ρ = +1.00, leave-one-out R² = 0.78) by
each group's Chinese-language Wikipedia prevalence. The salience gradient
persists through full instruction-tuning and RLHF, and the apparent
Chinese-origin advantage dissolves under per-group stratification.

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{iconicgroupbias2026,
  title   = {Iconic-Group Bias in Vision-Language Models: How Salience
             Gradients Render Chinese Ethnic Minority Cultures Invisible},
  author  = {[AUTHORS]},
  journal = {AI \& Society},
  year    = {2026},
  doi     = {10.XXXX/XXXXXX}
}
```

And cite the data deposit:

```bibtex
@dataset{iconicgroupbias_data,
  title     = {Iconic-Group Bias in Vision-Language Models: predictions,
               metrics, and salience snapshots},
  author    = {[AUTHORS]},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```

## Repository layout

```
.
├── README.md                       this file
├── LICENSE                         MIT — covers all code
├── LICENSE.data                    CC-BY-4.0 — covers derived data (predictions, metrics, ratings)
├── CITATION.cff                    machine-readable citation metadata
├── environment.yml                 conda environment used for the experiments
│
├── configs/
│   ├── models.yaml                 model registry and HF / API endpoints
│   ├── prompts.yaml                Task 1 / Task 2 / shuffled-MCQ / neutral-prompt templates
│   ├── evaluation.yaml             evaluation knobs
│   └── cultural_vocabulary.yaml    seed vocabulary for the CTC metric
│
├── data/
│   ├── DATASHEET.md                Gebru-style datasheet for the benchmark
│   ├── metadata.csv                image_id, group, source URL, SHA-256, pHash, EXIF stats
│   └── dataset_audit.csv           perceptual-hash dedup audit (no cross-group near-duplicates)
│
├── salience/
│   └── wikipedia_snapshot_*.csv    six-proxy Wikipedia snapshot, dated
│
├── results/
│   ├── raw/                        per-image predictions (10 files Task 1 + 10 files Task 2)
│   ├── metrics/                    aggregated CSVs and JSONs (37 files)
│   ├── human_eval/                 150 expert ratings, split-half reliability report
│   └── figures/                    rendered PDFs of every paper figure
│
├── scripts/                        every analysis script invoked end-to-end
│   ├── run_classification.py       Task 1 inference driver
│   ├── run_description.py          Task 2 inference driver
│   ├── compute_classification_metrics.py
│   ├── compute_description_metrics.py
│   ├── run_statistical_tests.py    OBI / LES / bootstrap CIs / McNemar
│   ├── salience_prevalence_test.py
│   ├── salience_prevalence_robust.py
│   ├── analyze_text_confusion.py
│   ├── analyze_position_bias.py    (Appendix C: shuffled MCQ)
│   ├── analyze_prompt_leak.py      (Appendix D: neutral prompt)
│   ├── analyze_rater_calibration.py
│   ├── analyze_deep_dive.py        (Appendix E core)
│   ├── analyze_deep_dive_part2.py  (Appendix E iconicness regression + tokenizer probe)
│   ├── audit_dataset.py            perceptual-hash audit + datasheet generator
│   └── generate_figures.py
│
└── src/                            reusable model-wrapper and inference modules
    ├── models/                     BaseVLM + per-model implementations
    ├── inference/                  task1_classification.py, task2_description.py
    ├── evaluation/                 metrics.py
    ├── visualization/
    └── utils/
```

## Quickstart — reproduce the headline finding (10 minutes, CPU only)

The full inference pipeline requires GPU + API keys, but the core empirical
claim of the paper (the ρ = +1.00 salience correlation) can be reproduced from
the released per-image predictions and Wikipedia snapshot in ten minutes on a
laptop:

```bash
git clone https://github.com/[user]/vlm-iconic-group-bias.git
cd vlm-iconic-group-bias

# Create the conda environment
conda env create -f environment.yml
conda activate vlm_eval

# Reproduce the salience correlation (Spearman ρ, exact permutation p,
# Holm-Bonferroni across six proxies, jackknife LOO ρ, log-linear LOO R²)
python scripts/salience_prevalence_test.py
python scripts/salience_prevalence_robust.py

# Reproduce the asymmetric-confusion table (Table 4 in the paper)
# and the per-image n=30,000 iconicness regression
python scripts/analyze_deep_dive.py
python scripts/analyze_deep_dive_part2.py

# Regenerate every figure
python scripts/generate_figures.py
```

Outputs land under `results/metrics/` and `results/figures/`.

## Re-running inference (GPU + API keys required)

To re-run the full Task 1 / Task 2 inference from scratch you need:

- An NVIDIA GPU with ≥ 80 GB VRAM (we used a single H100) for the open-source
  models (Qwen2-VL-7B, Qwen2.5-VL-7B, LLaMA-3.2-Vision-11B). LLaMA-3.2-Vision
  requires accepting the gated repository on HuggingFace.
- API keys for OpenAI (`OPENAI_API_KEY`) and Anthropic (`ANTHROPIC_API_KEY`).
- The 3,000-image benchmark, reconstructed from `data/metadata.csv` (see
  *Reconstructing the image set* below).

Inference for the full grid took approximately 36 GPU-hours on a single H100
plus ~$15 USD in API spend. Per-model SLURM scripts are provided under
`slurm/` for convenience but are not required for non-cluster environments.

## Reconstructing the image set

The release omits the 3,000 source images for the reasons stated above. The
`data/metadata.csv` file contains the source URL, SHA-256 digest, and
perceptual-hash signature for every image. A simple recovery script
(`scripts/fetch_images.py`, supplied separately on request) downloads from
each source URL and verifies the hash; researchers can also choose to
substitute a community-curated alternative corpus that preserves the
five-group label scheme. We strongly recommend the latter for any
deployment-grade extension of this work.

## Hardware and software environment

- Python 3.10
- CUDA 12.1 (open-source models) — not required for re-running analysis from
  released predictions
- Single H100 80 GB GPU
- Conda environment specified in `environment.yml`
- TeX Live 2023 to recompile the paper

## Acknowledgements

Two domain experts in Chinese ethnic costume culture contributed the 150
human-evaluation ratings. See `Acknowledgements` in the paper for further
detail.

## Contact

Issues and questions: [open an issue on GitHub] / [author email after review].

## Licence

- **Code** in this repository: MIT (see `LICENSE`).
- **Derived data** in `data/`, `salience/`, `results/raw/`, `results/metrics/`,
  and `results/human_eval/`: CC-BY-4.0 (see `LICENSE.data`).
- **Source images** at the URLs listed in `data/metadata.csv`: copyright of
  the original publishers; not redistributed by this repository.

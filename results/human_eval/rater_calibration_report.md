# Rater Calibration Report

**Design**: Split-annotation. Two domain experts rated disjoint stratified subsets (aiming for ~75 items each) of the 150 human-evaluation descriptions, balanced across (ethnic group × model × language) cells.

**Rationale**: Budget constraints precluded overlapping annotation. Consequently, formal inter-rater reliability statistics (Krippendorff's α, Cohen's κ) cannot be computed. This is acknowledged as a limitation. In place of IRR we report rater scale calibration and cross-rater model-ranking consistency, plus split-half reliability on the combined dataset as a proxy.

## Scale Calibration (per-rater means)

### Rater `UNKNOWN` (n = 150 items)

| Dimension | n | Mean | SD | Median |
|---|---|---|---|---|
| Rating_1to5 | 0 | — | — | — |
| Accuracy_1to5 | 0 | — | — | — |
| Completeness_1to5 | 0 | — | — | — |
| Cultural_Depth_1to5 | 0 | — | — | — |

## Cross-Rater Rank Consistency

Spearman ρ on per-model means, computed within each rater's sub-sample.


## Split-Half Model-Ranking Reliability (Proxy)

For each dimension, the dataset is randomly halved 100 times; per-model means computed in each half; ρ between halves reported. This lower-bounds the consistency of the overall rating signal.


## Interpretation Guide
- Mean ρ ≥ 0.80: model ordering robust to rater sub-sample.
- Mean shift p ≥ 0.10: raters calibrated similarly.
- Split-half ρ ≥ 0.80: overall rating signal is stable.
- Any dimension falling below these thresholds should be flagged in the paper's §5.3 limitations.

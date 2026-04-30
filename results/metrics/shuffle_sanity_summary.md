# C3: MCQ-shuffle sanity-check results

Compares per-model Task 1 accuracy under the **fixed** MCQ option order (A:Miao, B:Dong, C:Yi, D:Li, E:Tibetan, as in the submitted manuscript) vs **per-image shuffled** option order.

The position-A fraction measures, under shuffled prompts, how often the model still outputs letter A. Random behaviour would give 0.20; a fraction >> 0.20 indicates positional bias that the fixed-order design conflates with content recognition.

| Model | Lang | n | Acc (fixed) | Acc (shuffle) | Δ | P(predict=A) | A | B | C | D | E |
|---|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-4.5 | en | 100 | 0.284 | 0.33 | 0.046 | 0.16 | 16 | 26 | 18 | 24 | 16 |
| claude-haiku-4.5 | zh | 100 | 0.33 | 0.26 | -0.07 | 0.07 | 7 | 27 | 25 | 21 | 20 |
| gpt-4o-mini | en | 100 | 0.351 | 0.37 | 0.019 | 0.22 | 22 | 23 | 18 | 19 | 18 |
| gpt-4o-mini | zh | 100 | 0.368 | 0.35 | -0.018 | 0.21 | 21 | 22 | 19 | 17 | 21 |
| llama-3.2-vision-11b | en | 100 | 0.27 | 0.31 | 0.04 | 0.23 | 23 | 22 | 22 | 16 | 17 |
| llama-3.2-vision-11b | zh | 100 | 0.378 | 0.43 | 0.052 | 0.37 | 37 | 24 | 22 | 14 | 3 |
| qwen2-vl-7b | en | 100 | 0.413 | 0.42 | 0.007 | 0.26 | 26 | 22 | 16 | 20 | 16 |
| qwen2-vl-7b | zh | 100 | 0.381 | 0.37 | -0.011 | 0.23 | 23 | 23 | 20 | 18 | 16 |
| qwen2.5-vl-7b | en | 100 | 0.395 | 0.41 | 0.015 | 0.22 | 22 | 23 | 17 | 19 | 19 |
| qwen2.5-vl-7b | zh | 100 | 0.394 | 0.39 | -0.004 | 0.2 | 20 | 23 | 14 | 20 | 23 |

## Per-group comparison (shuffled vs fixed)

### claude-haiku-4.5 / en

| Group | Fixed | Shuffled | Δ |
|---|---|---|---|
| Miao | 0.678 | 0.85 | 0.172 |
| Dong | 0.26 | 0.1 | -0.16 |
| Yi | 0.25 | 0.3 | 0.05 |
| Li | 0.022 | 0.1 | 0.078 |
| Tibetan | 0.21 | 0.3 | 0.09 |

### claude-haiku-4.5 / zh

| Group | Fixed | Shuffled | Δ |
|---|---|---|---|
| Miao | 0.275 | 0.7 | 0.425 |
| Dong | 0.545 | 0.05 | -0.495 |
| Yi | 0.438 | 0.25 | -0.188 |
| Li | 0.027 | 0.1 | 0.073 |
| Tibetan | 0.365 | 0.2 | -0.165 |

### gpt-4o-mini / en

| Group | Fixed | Shuffled | Δ |
|---|---|---|---|
| Miao | 0.988 | 1.0 | 0.012 |
| Dong | 0.005 | 0.0 | -0.005 |
| Yi | 0.095 | 0.1 | 0.005 |
| Li | 0.003 | 0.0 | -0.003 |
| Tibetan | 0.665 | 0.75 | 0.085 |

### gpt-4o-mini / zh

| Group | Fixed | Shuffled | Δ |
|---|---|---|---|
| Miao | 0.993 | 1.0 | 0.007 |
| Dong | 0.013 | 0.0 | -0.013 |
| Yi | 0.05 | 0.0 | -0.05 |
| Li | 0.01 | 0.0 | -0.01 |
| Tibetan | 0.773 | 0.75 | -0.023 |

### llama-3.2-vision-11b / en

| Group | Fixed | Shuffled | Δ |
|---|---|---|---|
| Miao | 0.997 | 1.0 | 0.003 |
| Dong | 0.0 | 0.0 | 0.0 |
| Yi | 0.007 | 0.0 | -0.007 |
| Li | 0.0 | 0.0 | 0.0 |
| Tibetan | 0.345 | 0.55 | 0.205 |

### llama-3.2-vision-11b / zh

| Group | Fixed | Shuffled | Δ |
|---|---|---|---|
| Miao | 0.863 | 0.7 | -0.163 |
| Dong | 0.065 | 0.15 | 0.085 |
| Yi | 0.417 | 0.4 | -0.017 |
| Li | 0.0 | 0.0 | 0.0 |
| Tibetan | 0.547 | 0.9 | 0.353 |

### qwen2-vl-7b / en

| Group | Fixed | Shuffled | Δ |
|---|---|---|---|
| Miao | 0.995 | 0.9 | -0.095 |
| Dong | 0.012 | 0.05 | 0.038 |
| Yi | 0.137 | 0.2 | 0.063 |
| Li | 0.025 | 0.05 | 0.025 |
| Tibetan | 0.897 | 0.9 | 0.003 |

### qwen2-vl-7b / zh

| Group | Fixed | Shuffled | Δ |
|---|---|---|---|
| Miao | 0.982 | 0.9 | -0.082 |
| Dong | 0.067 | 0.05 | -0.017 |
| Yi | 0.073 | 0.05 | -0.023 |
| Li | 0.035 | 0.05 | 0.015 |
| Tibetan | 0.748 | 0.8 | 0.052 |

### qwen2.5-vl-7b / en

| Group | Fixed | Shuffled | Δ |
|---|---|---|---|
| Miao | 0.995 | 1.0 | 0.005 |
| Dong | 0.025 | 0.05 | 0.025 |
| Yi | 0.042 | 0.05 | 0.008 |
| Li | 0.018 | 0.05 | 0.032 |
| Tibetan | 0.893 | 0.9 | 0.007 |

### qwen2.5-vl-7b / zh

| Group | Fixed | Shuffled | Δ |
|---|---|---|---|
| Miao | 0.475 | 0.8 | 0.325 |
| Dong | 0.368 | 0.1 | -0.268 |
| Yi | 0.098 | 0.05 | -0.048 |
| Li | 0.068 | 0.05 | -0.018 |
| Tibetan | 0.96 | 0.95 | -0.01 |

## Decision rule
- If Δ accuracy is small (|Δ| < 0.05) and position_A_fraction is close to 0.20, the paper's headline findings (mode collapse onto Miao/Tibetan; Dong/Yi/Li near floor) are robust to positional bias.
- If Δ accuracy is large (> 0.10) or position_A_fraction deviates substantially from 0.20, the paper must acknowledge positional bias as a co-contributor to the observed mode collapse, and the shuffled-order numbers should be reported as the primary result.

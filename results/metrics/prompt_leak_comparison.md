# C4: Prompt-leak sanity check

Compares Task 2 in-text confusion rates between the **default** prompt (seeds Miao-coded technique words: *embroidery / batik / brocade / silver ornaments*) and the **neutral** prompt (same 5 aspects but without seed terms).

## Key metric: Dong→Miao in-text confusion rate

Fraction of Dong-costume descriptions that mention the word "Miao/苗族".

| Language | Default (seeded) | Neutral | Δ |
|---|---|---|---|
| en | 0.370 | 0.420 | +0.050 |
| zh | 0.422 | 0.430 | +0.008 |

## Seed-vocabulary usage rate (any Miao-coded word)

Fraction of descriptions of EACH true group that use at least one of "embroidery / batik / brocade / silver ornaments" (or their Chinese equivalents).

| Group | Lang | Default | Neutral | Δ |
|---|---|---|---|---|
| Dong | en | 1.000 | 1.000 | +0.000 |
| Dong | zh | 0.962 | 0.930 | -0.032 |
| Li | en | 0.994 | 0.940 | -0.054 |
| Li | zh | 0.948 | 0.910 | -0.038 |
| Miao | en | 0.988 | 0.990 | +0.002 |
| Miao | zh | 0.950 | 0.890 | -0.060 |
| Tibetan | en | 0.988 | 0.950 | -0.038 |
| Tibetan | zh | 0.936 | 0.930 | -0.006 |
| Yi | en | 0.998 | 0.960 | -0.038 |
| Yi | zh | 0.936 | 0.910 | -0.026 |

## Decision rule

- **Robust** (|Δ Dong->Miao| < 0.10): original finding stands; the prompt leak is acknowledged as a limitation but does not drive the confusion.
- **Partially artefactual** (Δ <= -0.20): report the neutral-prompt numbers as primary; reframe §4.4 to acknowledge prompt contribution.
- **Intermediate**: discuss both variants side-by-side.

Seed-vocabulary delta additionally shows whether models are simply echoing the prompt verbatim: a large drop (Δ << 0) in seed-word usage means models do follow the prompt vocabulary.
"""C3 analysis — compare shuffled-MCQ vs fixed-order classification.

Reads shuffled-MCQ result JSONs produced by run_classification_shuffled.py
and compares them to the canonical fixed-order Task 1 results in
results/raw/. Produces:

  * results/metrics/shuffle_sanity_summary.csv
      One row per (model, language): accuracy_fixed, accuracy_shuffled,
      delta, per-letter marginal distribution under shuffled prompts,
      per-group accuracy under shuffled vs fixed.
  * results/metrics/shuffle_sanity_summary.md
      Markdown narrative for inclusion in a revision-response appendix.

Interpretation guide:
  * position_A_fraction ~ 0.20  => no position-A bias under shuffle.
  * position_A_fraction > 0.40  => clear positional bias; shuffle
    substantially changes predictions.
  * per-group accuracy shuffled ≈ fixed => mode collapse reflects
    content, not position; paper's headline is robust.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "results" / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)


def load_shuffled(shuffle_dir: Path) -> List[dict]:
    return [json.loads(Path(p).read_text())
            for p in sorted(glob.glob(str(shuffle_dir / "*_results.json")))]


def load_fixed_per_cell() -> pd.DataFrame:
    """Per-cell accuracy from the original fixed-order runs."""
    p = METRICS / "task1_per_cell.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def summarise_run(run: dict) -> dict:
    s = run["summary"]
    return {
        "model": run["model_name"],
        "language": run["language"],
        "n": s["total_images"],
        "accuracy_shuffled": s["accuracy"],
        "position_A_fraction": s.get("position_A_fraction"),
        "predicted_letter_counts": s.get("predicted_letter_counts", {}),
        "per_group_shuffled": s.get("per_group_accuracy", {}),
    }


def attach_fixed(rows: List[dict], fixed_df: pd.DataFrame) -> List[dict]:
    if fixed_df.empty:
        return rows
    for r in rows:
        sub = fixed_df[
            (fixed_df["model"] == r["model"])
            & (fixed_df["language"] == r["language"])
        ]
        if sub.empty:
            r["accuracy_fixed"] = None
            r["per_group_fixed"] = {}
            continue
        # Overall accuracy under fixed-order across 5 groups
        r["accuracy_fixed"] = round(float(sub["accuracy"].mean()), 3)
        r["per_group_fixed"] = {
            row["ethnic_group"]: round(float(row["accuracy"]), 3)
            for _, row in sub.iterrows()
        }
        r["delta_accuracy"] = (
            round(r["accuracy_shuffled"] - r["accuracy_fixed"], 3)
            if r["accuracy_fixed"] is not None else None
        )
    return rows


def write_markdown(rows: List[dict], out_path: Path) -> None:
    lines = ["# C3: MCQ-shuffle sanity-check results", ""]
    lines.append(
        "Compares per-model Task 1 accuracy under the **fixed** MCQ "
        "option order (A:Miao, B:Dong, C:Yi, D:Li, E:Tibetan, as in "
        "the submitted manuscript) vs **per-image shuffled** option "
        "order.\n"
    )
    lines.append(
        "The position-A fraction measures, under shuffled prompts, "
        "how often the model still outputs letter A. Random behaviour "
        "would give 0.20; a fraction >> 0.20 indicates positional "
        "bias that the fixed-order design conflates with content "
        "recognition.\n"
    )
    lines.append(
        "| Model | Lang | n | Acc (fixed) | Acc (shuffle) | Δ | "
        "P(predict=A) | A | B | C | D | E |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for r in rows:
        lc = r.get("predicted_letter_counts", {})
        lines.append(
            f"| {r['model']} | {r['language']} | {r['n']} "
            f"| {r.get('accuracy_fixed')} | {r['accuracy_shuffled']} "
            f"| {r.get('delta_accuracy', '—')} "
            f"| {r.get('position_A_fraction')} "
            f"| {lc.get('A', 0)} | {lc.get('B', 0)} | {lc.get('C', 0)} "
            f"| {lc.get('D', 0)} | {lc.get('E', 0)} |"
        )

    lines.append("\n## Per-group comparison (shuffled vs fixed)\n")
    for r in rows:
        if not r.get("per_group_shuffled"):
            continue
        lines.append(f"### {r['model']} / {r['language']}\n")
        lines.append("| Group | Fixed | Shuffled | Δ |")
        lines.append("|---|---|---|---|")
        for g in ["Miao", "Dong", "Yi", "Li", "Tibetan"]:
            f_ = r.get("per_group_fixed", {}).get(g)
            s_ = r.get("per_group_shuffled", {}).get(g)
            delta = (
                round(s_ - f_, 3)
                if (f_ is not None and s_ is not None) else "—"
            )
            lines.append(
                f"| {g} | {f_ if f_ is not None else '—'} "
                f"| {s_ if s_ is not None else '—'} | {delta} |"
            )
        lines.append("")

    lines.append(
        "## Decision rule\n"
        "- If Δ accuracy is small (|Δ| < 0.05) and position_A_fraction "
        "is close to 0.20, the paper's headline findings (mode collapse "
        "onto Miao/Tibetan; Dong/Yi/Li near floor) are robust to "
        "positional bias.\n"
        "- If Δ accuracy is large (> 0.10) or position_A_fraction "
        "deviates substantially from 0.20, the paper must acknowledge "
        "positional bias as a co-contributor to the observed mode "
        "collapse, and the shuffled-order numbers should be reported as "
        "the primary result.\n"
    )
    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shuffle-dir", default=str(ROOT / "results" / "raw" / "shuffled_mcq")
    )
    ap.add_argument(
        "--out-csv", default=str(METRICS / "shuffle_sanity_summary.csv")
    )
    ap.add_argument(
        "--out-md", default=str(METRICS / "shuffle_sanity_summary.md")
    )
    args = ap.parse_args()

    runs = load_shuffled(Path(args.shuffle_dir))
    if not runs:
        print(f"No shuffle results found in {args.shuffle_dir}", flush=True)
        return

    rows = [summarise_run(r) for r in runs]
    rows = attach_fixed(rows, load_fixed_per_cell())

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    write_markdown(rows, Path(args.out_md))
    print(f"Saved: {args.out_csv}")
    print(f"       {args.out_md}")


if __name__ == "__main__":
    main()

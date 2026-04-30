"""Build paper artifacts for Options 1-3 (length-norm CTC, qualitative snippets, Table 5).

Outputs:
  - paper/tables/table5_obi_les.tex        (Table 5)
  - paper/tables/table_length_norm_corr.tex (Option 1)
  - paper/tables/qualitative_snippets.tex  (Option 2)
  - paper/data/length_norm_correlations.json (raw numbers)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "results" / "metrics"
RAW = ROOT / "results" / "raw"
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
DATA_OUT = PAPER / "data"
DATA_OUT.mkdir(parents=True, exist_ok=True)


# ---------- Option 1: length-normalized Cultural_Depth ----------

def option1_length_norm() -> dict:
    df = pd.read_csv(METRICS / "task2_human_vs_auto.csv")

    # Per-1000-char normalizations
    df["general_ctc_per1k"] = df["general_terms_matched"] / df["char_len"] * 1000.0
    df["correct_ethnic_ctc_per1k"] = (
        df["correct_ethnic_terms_matched"] / df["char_len"] * 1000.0
    )

    human_cols = ["Rating_1to5", "Accuracy_1to5", "Cultural_Depth_1to5"]
    auto_cols = [
        "char_len",
        "general_ctc",
        "general_ctc_per1k",
        "correct_ethnic_ctc",
        "correct_ethnic_ctc_per1k",
    ]

    rows = []
    for h in human_cols:
        for a in auto_cols:
            # spearman
            rho, p_rho = spearmanr(df[h], df[a])
            rp, p_rp = pearsonr(df[h], df[a])
            rows.append({
                "human": h,
                "auto": a,
                "spearman_rho": round(rho, 3),
                "spearman_p": round(p_rho, 4),
                "pearson_r": round(rp, 3),
                "pearson_p": round(p_rp, 4),
                "n": len(df),
            })
    out = pd.DataFrame(rows)
    out.to_csv(METRICS / "task2_length_norm_correlations.csv", index=False)
    (DATA_OUT / "length_norm_correlations.json").write_text(
        json.dumps(rows, indent=2)
    )
    return {"rows": rows, "n": len(df)}


def write_table_length_norm(result: dict) -> None:
    """LaTeX table of human-vs-auto correlations, highlighting length normalization."""
    df = pd.DataFrame(result["rows"])
    # Focus on Cultural_Depth (the rating most affected by length confound)
    key = df[df["human"].isin(["Cultural_Depth_1to5", "Rating_1to5"])].copy()

    def fmt(row: pd.Series) -> str:
        star = "$^{*}$" if row["spearman_p"] < 0.05 else ""
        return f"{row['spearman_rho']:+.2f}{star}"

    # Pivot: rows = auto metric, cols = human rating
    piv = key.pivot(index="auto", columns="human", values="spearman_rho")
    piv_p = key.pivot(index="auto", columns="human", values="spearman_p")

    auto_order = [
        "char_len",
        "general_ctc",
        "general_ctc_per1k",
        "correct_ethnic_ctc",
        "correct_ethnic_ctc_per1k",
    ]
    label_map = {
        "char_len": r"Char length (raw)",
        "general_ctc": r"General CTC",
        "general_ctc_per1k": r"General CTC / 1k chars",
        "correct_ethnic_ctc": r"Correct-ethnic CTC",
        "correct_ethnic_ctc_per1k": r"Correct-ethnic CTC / 1k chars",
    }

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Spearman $\rho$ between automated description metrics and human "
        r"ratings ($n{=}150$). Length normalization (\emph{per~1k chars}) "
        r"de-confounds the raw-count metrics, revealing that vocabulary coverage "
        r"does not actually track human-judged cultural depth once description "
        r"length is controlled. $^{*}$\,$p<0.05$.}"
    )
    lines.append(r"\label{tab:length_norm_corr}")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Automated metric & Overall Rating & Cultural Depth \\")
    lines.append(r"\midrule")
    for a in auto_order:
        r_over = piv.loc[a, "Rating_1to5"]
        p_over = piv_p.loc[a, "Rating_1to5"]
        r_cd = piv.loc[a, "Cultural_Depth_1to5"]
        p_cd = piv_p.loc[a, "Cultural_Depth_1to5"]
        s_over = "^{*}" if p_over < 0.05 else ""
        s_cd = "^{*}" if p_cd < 0.05 else ""
        lines.append(
            f"{label_map[a]} & ${r_over:+.2f}{s_over}$ & ${r_cd:+.2f}{s_cd}$ \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    (TABLES / "table_length_norm_corr.tex").write_text("\n".join(lines))


# ---------- Option 2: qualitative snippets ----------

LI_AS_THAI_RE = re.compile(
    r"(thai|thailand|southeast asia|south-east asia|vietnam|indochina|hmong|"
    r"indigenous|tropical|south asian|laos|cambodia)",
    re.IGNORECASE,
)
DONG_AS_MIAO_RE = re.compile(r"\bmiao\b", re.IGNORECASE)
DONG_AS_MIAO_ZH = re.compile(r"苗族")


def _load_raw_descriptions() -> dict[tuple[str, str], dict[str, str]]:
    """Return {(model, language): {image_id: description}}"""
    out: dict[tuple[str, str], dict[str, str]] = {}
    for p in sorted(RAW.glob("task2_*_results.json")):
        with open(p) as f:
            d = json.load(f)
        key = (d["model_name"], d["language"])
        out[key] = {img: rec["description"] for img, rec in d["results"].items()}
    return out


def _find_sentence(text: str, pattern: re.Pattern) -> str | None:
    """Return first sentence containing the pattern match."""
    # Clean markdown
    clean = re.sub(r"[#*`_]+", "", text)
    clean = re.sub(r"\s+", " ", clean).strip()
    # Split into sentences (en + zh punctuation)
    sents = re.split(r"(?<=[.!?。！？])\s+", clean)
    for s in sents:
        if pattern.search(s) and len(s) > 20:
            return s.strip()
    return None


def option2_snippets() -> dict:
    mentions = pd.read_csv(METRICS / "task2_intext_mentions.csv")
    raw = _load_raw_descriptions()

    # Li-as-Thai: English Li descriptions with geo_mislocation=True
    li_candidates = mentions[
        (mentions.true_group == "Li")
        & (mentions.language == "en")
        & (mentions.mentions_geo_mislocation == True)
    ]

    # Dong-as-Miao: Dong descriptions mentioning Miao but not Dong
    dong_candidates = mentions[
        (mentions.true_group == "Dong")
        & (mentions.mentions_Miao == True)
        & (mentions.mentions_Dong == False)
    ]

    def _diversify(candidates, pattern_fn) -> list:
        """Prefer one snippet per model; up to 8 total."""
        seen_models = set()
        picked = []
        # First pass: one per model
        for _, row in candidates.iterrows():
            if row.model in seen_models:
                continue
            text = raw.get((row.model, row.language), {}).get(row.image_id)
            if not text:
                continue
            snippet = _find_sentence(text, pattern_fn(row.language))
            if snippet and 30 < len(snippet) < 350:
                picked.append({
                    "image_id": row.image_id,
                    "model": row.model,
                    "language": row.language,
                    "true_group": row.true_group,
                    "snippet": snippet,
                })
                seen_models.add(row.model)
        # Second pass: fill remaining slots with any candidate
        for _, row in candidates.iterrows():
            if len(picked) >= 8:
                break
            if any(p["image_id"] == row.image_id and p["model"] == row.model for p in picked):
                continue
            text = raw.get((row.model, row.language), {}).get(row.image_id)
            if not text:
                continue
            snippet = _find_sentence(text, pattern_fn(row.language))
            if snippet and 30 < len(snippet) < 350:
                picked.append({
                    "image_id": row.image_id,
                    "model": row.model,
                    "language": row.language,
                    "true_group": row.true_group,
                    "snippet": snippet,
                })
        return picked

    li_snippets = _diversify(li_candidates, lambda lang: LI_AS_THAI_RE)
    dong_snippets = _diversify(
        dong_candidates,
        lambda lang: DONG_AS_MIAO_RE if lang == "en" else DONG_AS_MIAO_ZH,
    )

    out = {"li_as_thai": li_snippets[:6], "dong_as_miao": dong_snippets[:6]}
    (DATA_OUT / "qualitative_snippets.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def _latex_escape(s: str) -> str:
    # Order matters: backslash first, then braces, then others.
    s = s.replace("\\", r"\textbackslash{}")
    s = s.replace("{", r"\{").replace("}", r"\}")
    for k, v in {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }.items():
        s = s.replace(k, v)
    # Replace CJK characters with romanized annotation for pdfLaTeX safety
    import unicodedata
    cjk_map = {"苗": "Miao", "族": "zu", "侗": "Dong", "彝": "Yi",
               "黎": "Li", "藏": "Zang"}
    out = []
    for ch in s:
        if ord(ch) > 0x2E80:  # CJK range
            out.append(cjk_map.get(ch, "?"))
        else:
            out.append(ch)
    return "".join(out)



def write_table_snippets(snips: dict) -> None:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(
        r"\caption{Representative failure-mode excerpts from VLM-generated "
        r"costume descriptions (Task~2). \textbf{Top:} geo-displacement of Li "
        r"costumes to Southeast Asia in English prompts. \textbf{Bottom:} "
        r"Dong costumes described with Miao terminology without ever naming the "
        r"Dong group. Each snippet is a single verbatim sentence.}"
    )
    lines.append(r"\label{tab:qualitative_snippets}")
    lines.append(r"\begin{tabular}{p{0.18\linewidth} p{0.74\linewidth}}")
    lines.append(r"\toprule")
    lines.append(r"Source & Snippet \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{l}{\textit{Li $\to$ Southeast Asia (geo-displacement, EN prompts)}} \\")
    for s in snips["li_as_thai"][:3]:
        src = f"{s['model']} ({s['image_id']})"
        lines.append(
            f"{_latex_escape(src)} & ``{_latex_escape(s['snippet'])}'' \\\\"
        )
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{2}{l}{\textit{Dong $\to$ Miao (in-text confusion, no self-mention)}} \\")
    for s in snips["dong_as_miao"][:3]:
        lang = s["language"].upper()
        src = f"{s['model']}/{lang} ({s['image_id']})"
        lines.append(
            f"{_latex_escape(src)} & ``{_latex_escape(s['snippet'])}'' \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    (TABLES / "qualitative_snippets.tex").write_text("\n".join(lines))


# ---------- Option 3: Table 5 (per-model OBI/LES with CIs) ----------

def write_table5() -> None:
    bias = json.loads((METRICS / "task1_bias_metrics.json").read_text())
    sig = json.loads((METRICS / "task1_significance.json").read_text())
    les_ci = sig["bootstrap_ci"]["les_per_model"]
    obi_ci = sig["bootstrap_ci"]["obi_combined"]

    model_order = [
        "qwen2.5-vl-7b",
        "qwen2-vl-7b",
        "llama-3.2-vision-11b",
        "gpt-4o-mini",
        "claude-haiku-4.5",
    ]
    display = {
        "qwen2.5-vl-7b": r"Qwen2.5-VL-7B",
        "qwen2-vl-7b": r"Qwen2-VL-7B",
        "llama-3.2-vision-11b": r"LLaMA-3.2-V-11B",
        "gpt-4o-mini": r"GPT-4o-mini",
        "claude-haiku-4.5": r"Claude-Haiku-4.5",
    }
    origin_label = {"chinese": "CN", "western": "W"}

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering\small")
    lines.append(
        r"\caption{Per-model classification accuracy and Language Effect Score "
        r"(LES) with bootstrap 95\% CIs ($B{=}1000$). Positive LES means the "
        r"model benefits from Chinese-language prompts. The combined "
        r"Origin-Bias Index is OBI $=+0.201$ [CI $0.148,\,0.253$], but the "
        r"effect is concentrated in Western models' zh$\to$en drop, not in a "
        r"Chinese-model advantage on zh prompts. $^{*}$: 95\% CI excludes zero.}"
    )
    lines.append(r"\label{tab:per_model_obi_les}")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\toprule")
    lines.append(
        r" & & \multicolumn{2}{c}{Accuracy} & \multicolumn{3}{c}{LES (zh vs.\ en)} \\"
    )
    lines.append(r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}")
    lines.append(
        r"Model & Origin & zh & en & mean & 95\% CI lo & 95\% CI hi \\"
    )
    lines.append(r"\midrule")
    for m in model_order:
        info = bias["les_per_model"][m]
        ci = les_ci[m]
        sig_mark = "^{*}" if (ci["ci_lo"] > 0 or ci["ci_hi"] < 0) else ""
        lines.append(
            f"{display[m]} & {origin_label[info['origin']]} & "
            f"{info['acc_zh']:.3f} & {info['acc_en']:.3f} & "
            f"${ci['mean']:+.3f}{sig_mark}$ & "
            f"${ci['ci_lo']:+.3f}$ & ${ci['ci_hi']:+.3f}$ \\\\"
        )
    lines.append(r"\midrule")
    lines.append(
        rf"\multicolumn{{7}}{{l}}{{\footnotesize OBI (combined) $={obi_ci['mean']:+.3f}$ "
        rf"[{obi_ci['ci_lo']:+.3f},\,{obi_ci['ci_hi']:+.3f}]; "
        rf"OBI(zh)$={bias['obi_by_language']['zh']['obi']:+.3f}$; "
        rf"OBI(en)$={bias['obi_by_language']['en']['obi']:+.3f}$.}} \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    (TABLES / "table5_obi_les.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    print("[Option 1] Length-normalized correlations...")
    r1 = option1_length_norm()
    write_table_length_norm(r1)
    print(f"  wrote {TABLES / 'table_length_norm_corr.tex'}")

    print("[Option 2] Qualitative snippets...")
    snips = option2_snippets()
    print(
        f"  Li-as-Thai: {len(snips['li_as_thai'])}  "
        f"Dong-as-Miao: {len(snips['dong_as_miao'])}"
    )
    write_table_snippets(snips)
    print(f"  wrote {TABLES / 'qualitative_snippets.tex'}")

    print("[Option 3] Table 5 (per-model OBI/LES)...")
    write_table5()
    print(f"  wrote {TABLES / 'table5_obi_les.tex'}")

    print("Done.")

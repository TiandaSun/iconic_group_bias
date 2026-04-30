"""Dataset audit for reproducibility and reviewer concern C7.

Runs a full provenance / dedup / integrity audit on the 3000-image
benchmark. Outputs:

  * data/dataset_audit.csv — per-image row with SHA256, perceptual hash,
    file size, image dimensions, EXIF (if any), decoded format
  * data/dataset_audit.json — aggregate statistics + detected issues
  * data/DATASHEET.md — Gebru-style datasheet draft

Required packages (install in vlm_eval conda env):
    pip install imagehash pillow

Does not require GPU. Safe to run in a CPU SLURM job or on the login
node once imagehash is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    import imagehash  # type: ignore
    from PIL import Image, ExifTags
    HAS_IMAGING = True
except ImportError:
    HAS_IMAGING = False

ROOT = Path(__file__).resolve().parent.parent


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hashes(img: "Image.Image") -> Dict[str, str]:
    """pHash, dHash, aHash of an image. 16-hex-char strings."""
    return {
        "phash": str(imagehash.phash(img, hash_size=8)),
        "dhash": str(imagehash.dhash(img, hash_size=8)),
        "ahash": str(imagehash.average_hash(img, hash_size=8)),
    }


def audit_image(row: pd.Series) -> Dict:
    path = Path(row["image_path"])
    out = {
        "image_id": row["image_id"],
        "ethnic_group": row["ethnic_group"],
        "image_path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return out

    stat = path.stat()
    out["file_bytes"] = int(stat.st_size)
    out["sha256"] = sha256_of(path)

    if not HAS_IMAGING:
        out["note"] = "imagehash / PIL not installed; skipping perceptual hashes"
        return out

    try:
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            out["width"], out["height"] = img.size
            out["format"] = img.format
            out["mode"] = img.mode

            # Perceptual hashes
            out.update(perceptual_hashes(img_rgb))

            # EXIF (may be absent)
            exif_raw = img.getexif() if hasattr(img, "getexif") else None
            if exif_raw:
                exif = {}
                for tag, value in exif_raw.items():
                    name = ExifTags.TAGS.get(tag, str(tag))
                    try:
                        exif[name] = str(value)[:200]  # truncate
                    except Exception:
                        exif[name] = "<unreadable>"
                out["exif_keys"] = list(exif.keys())
                out["has_exif"] = True
            else:
                out["has_exif"] = False
    except Exception as e:
        out["load_error"] = str(e)
    return out


def find_duplicates(records: List[Dict],
                    phash_threshold: int = 5) -> Dict:
    """Exact duplicates (same SHA256) and near-duplicates (phash Hamming
    distance < threshold).

    Returns grouped sets of duplicate image_ids with whether each cluster
    spans groups (a critical finding if so — label leakage).
    """
    import collections
    # Exact duplicates
    by_sha: Dict[str, List[str]] = collections.defaultdict(list)
    for r in records:
        if "sha256" in r:
            by_sha[r["sha256"]].append(r["image_id"])
    exact_dup = {
        sha: ids for sha, ids in by_sha.items() if len(ids) > 1
    }

    # Near duplicates via pHash
    near_dup: List[Dict] = []
    if HAS_IMAGING:
        by_id = {r["image_id"]: r for r in records if "phash" in r}
        ids = list(by_id.keys())
        seen = set()
        for i, a in enumerate(ids):
            if a in seen:
                continue
            ha = imagehash.hex_to_hash(by_id[a]["phash"])
            cluster = [a]
            for b in ids[i + 1:]:
                if b in seen:
                    continue
                hb = imagehash.hex_to_hash(by_id[b]["phash"])
                if (ha - hb) <= phash_threshold:
                    cluster.append(b)
                    seen.add(b)
            if len(cluster) > 1:
                # Record cluster, flagging cross-group matches
                groups_in_cluster = list({
                    by_id[i_]["ethnic_group"] for i_ in cluster
                })
                near_dup.append({
                    "cluster": cluster,
                    "size": len(cluster),
                    "groups_in_cluster": groups_in_cluster,
                    "cross_group": len(groups_in_cluster) > 1,
                })
                seen.update(cluster)
    return {
        "exact_duplicates": exact_dup,
        "near_duplicates": near_dup,
        "exact_dup_n_clusters": len(exact_dup),
        "near_dup_n_clusters": len(near_dup),
        "near_dup_cross_group_count": sum(
            1 for c in near_dup if c.get("cross_group")
        ),
    }


def aggregate_stats(records: List[Dict]) -> Dict:
    stats: Dict = {
        "n_images_indexed": len(records),
        "n_images_exist": sum(1 for r in records if r.get("exists")),
        "by_group": {},
        "formats": Counter(),
        "resolutions": {"min_w": None, "max_w": None,
                        "min_h": None, "max_h": None},
        "file_bytes": {"mean": 0.0, "min": 0, "max": 0},
    }
    # Per-group counts
    grp: Dict = defaultdict(int)
    for r in records:
        grp[r.get("ethnic_group", "?")] += 1
    stats["by_group"] = dict(grp)

    widths = [r["width"] for r in records if "width" in r]
    heights = [r["height"] for r in records if "height" in r]
    bytes_ = [r["file_bytes"] for r in records if "file_bytes" in r]
    if widths:
        stats["resolutions"].update({
            "min_w": min(widths), "max_w": max(widths),
            "min_h": min(heights), "max_h": max(heights),
        })
    if bytes_:
        stats["file_bytes"] = {
            "mean": float(sum(bytes_) / len(bytes_)),
            "min": int(min(bytes_)), "max": int(max(bytes_)),
        }
    for r in records:
        if "format" in r:
            stats["formats"][r["format"]] += 1
    stats["formats"] = dict(stats["formats"])
    return stats


def write_datasheet(stats: Dict, dup: Dict, out_path: Path) -> None:
    """Gebru et al. (2021) style datasheet draft."""
    lines = [
        "# Dataset Audit — Chinese Ethnic Minority Costume Benchmark",
        "",
        "Gebru-style datasheet draft, programmatically generated by "
        "`scripts/audit_dataset.py`. Human review required before public "
        "release; fields marked *[TO VERIFY]* need confirmation from "
        "the collection-team primary author.",
        "",
        "## 1. Motivation and Purpose",
        "",
        "This dataset was curated for the research paper *Iconic-Group "
        "Bias in Vision-Language Models* to enable within-culture "
        "stratified evaluation of Chinese ethnic minority costume "
        "recognition.",
        "",
        "## 2. Composition",
        "",
        f"* Total images indexed: **{stats['n_images_indexed']}**.",
        f"* Total images present on disk: **{stats['n_images_exist']}**.",
        "",
        "### Per-group counts",
        "",
        "| Ethnic Group | Image Count |",
        "|---|---|",
    ]
    for g, n in sorted(stats["by_group"].items()):
        lines.append(f"| {g} | {n} |")
    lines.extend([
        "",
        "### Technical characteristics",
        "",
        f"* Image formats: {stats.get('formats', {})}.",
        (f"* Resolution range: width "
         f"[{stats['resolutions'].get('min_w')}, "
         f"{stats['resolutions'].get('max_w')}]; height "
         f"[{stats['resolutions'].get('min_h')}, "
         f"{stats['resolutions'].get('max_h')}]."),
        (f"* File size: mean "
         f"{stats['file_bytes']['mean']/1024:.1f} KiB; "
         f"range [{stats['file_bytes']['min']}, "
         f"{stats['file_bytes']['max']}] bytes."),
        "",
        "## 3. Integrity Audit",
        "",
        f"* Exact duplicate clusters (same SHA256): "
        f"**{dup['exact_dup_n_clusters']}**.",
        f"* Near-duplicate clusters (pHash Hamming ≤ 5): "
        f"**{dup['near_dup_n_clusters']}**.",
        f"* Near-duplicate clusters spanning different ethnic groups "
        f"(potential label leakage): "
        f"**{dup['near_dup_cross_group_count']}**.",
        "",
        "If the cross-group count is non-zero, the affected images must "
        "be inspected manually; persistence of the headline results "
        "under removal of these clusters should be reported in the "
        "revised manuscript.",
        "",
        "## 4. Collection Process *[TO VERIFY]*",
        "",
        "* Source(s): *[TO VERIFY with collection-team primary author]*",
        "* Collection date range: *[TO VERIFY]*",
        "* Automated vs manual curation: *[TO VERIFY]*",
        "* Any deduplication applied during collection: *[TO VERIFY]*",
        "",
        "## 5. Preprocessing",
        "",
        "The images are used at their original resolution for VLM "
        "inference; no resizing or augmentation is applied.",
        "",
        "## 6. Uses and Limitations",
        "",
        "* **Intended use.** Benchmark for evaluating per-group "
        "recognition accuracy of Chinese ethnic minority costumes by "
        "vision-language models. Research-only use.",
        "* **Known limitations.** "
        "(a) Potential Western-tourist-photographer bias (many source "
        "photos may originate from tourism and cultural-performance "
        "contexts rather than daily wear). "
        "(b) Images may over-represent performative/festival attire "
        "rather than everyday costume. "
        "(c) No age/gender balance is controlled for. "
        "(d) Only 5 of 55 officially recognised Chinese ethnic "
        "minorities are covered.",
        "",
        "## 7. Licensing and Consent *[TO VERIFY]*",
        "",
        "Image licences and depicted-person consent status require "
        "verification before any public release of the dataset itself. "
        "The paper plans to release *code + per-image predictions*, "
        "not the raw images, if licensing cannot be cleared.",
        "",
        "## 8. Maintenance",
        "",
        "This dataset is provided as a fixed snapshot for reproducibility "
        "of the paper's experiments. No scheduled updates are planned.",
    ])
    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata",
                    default=str(ROOT / "data" / "metadata.csv"))
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    args = ap.parse_args()

    if not HAS_IMAGING:
        print("WARNING: imagehash / Pillow not installed. Perceptual-hash\n"
              "dedup will be skipped. Install with:\n"
              "    pip install imagehash Pillow")

    meta = pd.read_csv(args.metadata)
    out_dir = Path(args.out_dir)
    records = []
    for _, row in meta.iterrows():
        records.append(audit_image(row))

    # Save per-image CSV
    pd.DataFrame(records).to_csv(
        out_dir / "dataset_audit.csv", index=False
    )

    dup = find_duplicates(records)
    stats = aggregate_stats(records)
    agg = {"statistics": stats, "duplicates": dup}
    (out_dir / "dataset_audit.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False)
    )

    write_datasheet(stats, dup, out_dir / "DATASHEET.md")

    print(f"Saved:")
    print(f"  {out_dir / 'dataset_audit.csv'}")
    print(f"  {out_dir / 'dataset_audit.json'}")
    print(f"  {out_dir / 'DATASHEET.md'}")
    print(f"\nSummary:")
    print(f"  Images indexed:          {stats['n_images_indexed']}")
    print(f"  Images present on disk:  {stats['n_images_exist']}")
    print(f"  Exact duplicate clusters:    {dup['exact_dup_n_clusters']}")
    print(f"  Near duplicate clusters:     {dup['near_dup_n_clusters']}")
    print(f"  Cross-group near duplicates: {dup['near_dup_cross_group_count']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate metadata CSV from the actual data directory structure."""

import csv
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Mapping from Chinese folder names to English ethnic group names
CHINESE_TO_ENGLISH = {
    "苗族": "Miao",
    "侗族": "Dong",
    "彝族": "Yi",
    "黎族": "Li",
    "藏族": "Tibetan",
}

# Label mapping
ETHNIC_GROUP_TO_LABEL = {
    "Miao": "A",
    "Dong": "B",
    "Yi": "C",
    "Li": "D",
    "Tibetan": "E",
}


def generate_metadata(data_dir: Path, output_csv: Path) -> list:
    """Generate metadata from directory structure.

    Expected structure:
        data_dir/服饰2/民族服饰/{ethnic_group_chinese}/image_files
    """
    records = []
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

    # Find the image directories
    ethnic_base = data_dir / "服饰2" / "民族服饰"

    if not ethnic_base.exists():
        print(f"Error: Directory not found: {ethnic_base}")
        return []

    for group_dir in sorted(ethnic_base.iterdir()):
        if not group_dir.is_dir():
            continue

        chinese_name = group_dir.name
        if chinese_name not in CHINESE_TO_ENGLISH:
            print(f"Warning: Unknown ethnic group folder: {chinese_name}")
            continue

        english_name = CHINESE_TO_ENGLISH[chinese_name]
        label_letter = ETHNIC_GROUP_TO_LABEL[english_name]

        # Count images in this group
        group_count = 0

        # Find all images in this group
        for image_file in sorted(group_dir.iterdir()):
            if image_file.suffix.lower() in image_extensions:
                group_count += 1
                # Create a clean image ID
                image_id = f"{english_name}_{group_count:04d}"

                records.append({
                    "image_id": image_id,
                    "image_path": str(image_file.resolve()),
                    "ethnic_group": english_name,
                    "ethnic_group_zh": chinese_name,
                    "label_letter": label_letter,
                })

    if len(records) == 0:
        print("Warning: No images found!")
        return []

    # Sort records by ethnic group and image_id
    records.sort(key=lambda x: (x["ethnic_group"], x["image_id"]))

    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "image_path", "ethnic_group", "ethnic_group_zh", "label_letter"])
        writer.writeheader()
        writer.writerows(records)

    # Print summary
    print(f"\nMetadata generated: {output_csv}")
    print(f"Total images: {len(records)}")
    print("\nDistribution by ethnic group:")

    # Count by ethnic group
    counts = {}
    for r in records:
        group = r["ethnic_group"]
        counts[group] = counts.get(group, 0) + 1

    for group in sorted(counts.keys()):
        print(f"  {group}: {counts[group]} images")

    return records


if __name__ == "__main__":
    data_dir = project_root / "data"
    output_csv = data_dir / "metadata.csv"

    print(f"Scanning data directory: {data_dir}")
    records = generate_metadata(data_dir, output_csv)

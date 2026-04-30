#!/usr/bin/env python3
"""Create a small test metadata file for quick testing."""

import csv
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_test_metadata(
    full_metadata_path: Path,
    output_path: Path,
    samples_per_group: int = 5,
) -> int:
    """Create a small test metadata file.

    Args:
        full_metadata_path: Path to full metadata.csv.
        output_path: Path for output test metadata.
        samples_per_group: Number of samples per ethnic group.

    Returns:
        Total number of test samples.
    """
    # Read full metadata
    with open(full_metadata_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Group by ethnic_group
    groups = {}
    for row in rows:
        group = row['ethnic_group']
        if group not in groups:
            groups[group] = []
        groups[group].append(row)

    # Sample from each group
    test_rows = []
    for group, group_rows in sorted(groups.items()):
        sample = group_rows[:samples_per_group]
        test_rows.extend(sample)
        print(f"  {group}: {len(sample)} samples")

    # Write test metadata
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['image_id', 'image_path', 'ethnic_group', 'ethnic_group_zh', 'label_letter']

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in test_rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})

    print(f"\nTest metadata created: {output_path}")
    print(f"Total test samples: {len(test_rows)}")

    return len(test_rows)


if __name__ == "__main__":
    data_dir = project_root / "data"
    full_metadata = data_dir / "metadata.csv"
    test_metadata = data_dir / "metadata_test.csv"

    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    print(f"Creating test metadata with {samples} samples per ethnic group...")
    create_test_metadata(full_metadata, test_metadata, samples_per_group=samples)

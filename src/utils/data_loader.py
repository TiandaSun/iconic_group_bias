"""Data loading utilities for VLM evaluation."""

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Label mapping from ethnic group to letter
ETHNIC_GROUP_TO_LABEL = {
    "Miao": "A",
    "Dong": "B",
    "Yi": "C",
    "Li": "D",
    "Tibetan": "E",
}

# Reverse mapping
LABEL_TO_ETHNIC_GROUP = {v: k for k, v in ETHNIC_GROUP_TO_LABEL.items()}


def load_metadata(
    csv_path: Union[str, Path],
    image_base_dir: Optional[Union[str, Path]] = None,
    validate_paths: bool = False,
) -> pd.DataFrame:
    """Load image metadata from CSV file.

    Args:
        csv_path: Path to the metadata CSV file.
        image_base_dir: Base directory for image paths. If provided,
            image_path will be constructed as base_dir/relative_path.
        validate_paths: If True, check that all image paths exist.

    Returns:
        DataFrame with columns [image_id, image_path, ethnic_group, label_letter].

    Raises:
        FileNotFoundError: If CSV file doesn't exist.
        ValueError: If required columns are missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    logger.info(f"Loading metadata from {csv_path}")
    df = pd.read_csv(csv_path)

    # Standardize column names (handle common variations)
    column_mapping = {
        "id": "image_id",
        "ID": "image_id",
        "image_name": "image_id",
        "filename": "image_id",
        "path": "image_path",
        "file_path": "image_path",
        "image": "image_path",
        "ethnic": "ethnic_group",
        "ethnicity": "ethnic_group",
        "group": "ethnic_group",
        "class": "ethnic_group",
        "label": "ethnic_group",
        "category": "ethnic_group",
    }

    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Validate required columns
    required_cols = ["ethnic_group"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Generate image_id if not present
    if "image_id" not in df.columns:
        if "image_path" in df.columns:
            df["image_id"] = df["image_path"].apply(lambda x: Path(x).stem)
        else:
            df["image_id"] = [f"img_{i:05d}" for i in range(len(df))]

    # Handle image paths
    if "image_path" not in df.columns:
        # Try to construct from image_id
        if image_base_dir:
            # Look for images with common extensions
            base_dir = Path(image_base_dir)
            paths = []
            for img_id in df["image_id"]:
                found = False
                for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
                    potential_path = base_dir / f"{img_id}{ext}"
                    if potential_path.exists():
                        paths.append(str(potential_path))
                        found = True
                        break
                if not found:
                    paths.append(str(base_dir / f"{img_id}.jpg"))
            df["image_path"] = paths
        else:
            raise ValueError("image_path column missing and no image_base_dir provided")

    # Prepend base directory if provided
    if image_base_dir and "image_path" in df.columns:
        base_dir = Path(image_base_dir)
        df["image_path"] = df["image_path"].apply(
            lambda x: str(base_dir / x) if not Path(x).is_absolute() else x
        )

    # Standardize ethnic group names
    ethnic_group_mapping = {
        "miao": "Miao",
        "苗族": "Miao",
        "苗": "Miao",
        "dong": "Dong",
        "侗族": "Dong",
        "侗": "Dong",
        "yi": "Yi",
        "彝族": "Yi",
        "彝": "Yi",
        "li": "Li",
        "黎族": "Li",
        "黎": "Li",
        "tibetan": "Tibetan",
        "藏族": "Tibetan",
        "藏": "Tibetan",
        "zang": "Tibetan",
    }

    df["ethnic_group"] = df["ethnic_group"].apply(
        lambda x: ethnic_group_mapping.get(str(x).lower().strip(), x)
    )

    # Add label letter
    df["label_letter"] = df["ethnic_group"].map(ETHNIC_GROUP_TO_LABEL)

    # Validate labels
    invalid_groups = df[df["label_letter"].isna()]["ethnic_group"].unique()
    if len(invalid_groups) > 0:
        logger.warning(f"Unknown ethnic groups found: {invalid_groups}")

    # Validate image paths if requested
    if validate_paths:
        missing_images = []
        for idx, row in df.iterrows():
            if not Path(row["image_path"]).exists():
                missing_images.append(row["image_path"])

        if missing_images:
            logger.warning(f"Missing {len(missing_images)} images")
            if len(missing_images) <= 10:
                for path in missing_images:
                    logger.warning(f"  Missing: {path}")

    # Ensure consistent column order
    columns = ["image_id", "image_path", "ethnic_group", "label_letter"]
    extra_cols = [c for c in df.columns if c not in columns]
    df = df[columns + extra_cols]

    logger.info(
        f"Loaded {len(df)} images across {df['ethnic_group'].nunique()} ethnic groups"
    )
    logger.info(f"Distribution: {df['ethnic_group'].value_counts().to_dict()}")

    return df


def get_task1_batches(
    metadata: pd.DataFrame,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> Iterator[pd.DataFrame]:
    """Generate batches of images for Task 1 (classification).

    Args:
        metadata: DataFrame from load_metadata().
        batch_size: Number of images per batch.
        shuffle: Whether to shuffle before batching.
        seed: Random seed for shuffling (for reproducibility).

    Yields:
        DataFrame batches with batch_size rows each.
    """
    df = metadata.copy()

    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_batches = (len(df) + batch_size - 1) // batch_size
    logger.info(f"Creating {n_batches} batches of size {batch_size}")

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i : i + batch_size]
        yield batch


def get_image_paths_from_batch(batch: pd.DataFrame) -> List[str]:
    """Extract image paths from a batch DataFrame.

    Args:
        batch: DataFrame batch from get_task1_batches().

    Returns:
        List of image file paths.
    """
    return batch["image_path"].tolist()


def get_labels_from_batch(batch: pd.DataFrame) -> List[str]:
    """Extract ground truth labels from a batch DataFrame.

    Args:
        batch: DataFrame batch from get_task1_batches().

    Returns:
        List of label letters (A-E).
    """
    return batch["label_letter"].tolist()


def sample_task2_images(
    metadata: pd.DataFrame,
    n: int = 500,
    seed: int = 42,
    stratified: bool = True,
) -> pd.DataFrame:
    """Sample images for Task 2 (description generation).

    Args:
        metadata: DataFrame from load_metadata().
        n: Total number of images to sample.
        seed: Random seed for reproducibility.
        stratified: If True, sample equally from each ethnic group.

    Returns:
        DataFrame subset with sampled images.
    """
    if n >= len(metadata):
        logger.warning(f"Requested {n} samples but only {len(metadata)} available")
        return metadata.copy()

    if stratified:
        # Sample equally from each group
        groups = metadata["ethnic_group"].unique()
        n_per_group = n // len(groups)
        remainder = n % len(groups)

        samples = []
        for i, group in enumerate(sorted(groups)):
            group_df = metadata[metadata["ethnic_group"] == group]
            # Distribute remainder across first few groups
            group_n = n_per_group + (1 if i < remainder else 0)
            group_n = min(group_n, len(group_df))

            group_sample = group_df.sample(n=group_n, random_state=seed)
            samples.append(group_sample)

        sampled = pd.concat(samples, ignore_index=True)
        # Shuffle the combined result
        sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)

    else:
        sampled = metadata.sample(n=n, random_state=seed).reset_index(drop=True)

    logger.info(f"Sampled {len(sampled)} images for Task 2")
    logger.info(f"Distribution: {sampled['ethnic_group'].value_counts().to_dict()}")

    return sampled


def create_metadata_from_directory(
    image_dir: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Create metadata CSV from directory structure.

    Expects directory structure:
        image_dir/
            Miao/
                image1.jpg
                image2.jpg
            Dong/
                ...

    Args:
        image_dir: Root directory containing ethnic group subdirectories.
        output_csv: Optional path to save the generated CSV.

    Returns:
        DataFrame with image metadata.
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    records = []
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

    for group_dir in image_dir.iterdir():
        if not group_dir.is_dir():
            continue

        ethnic_group = group_dir.name

        for image_file in group_dir.iterdir():
            if image_file.suffix.lower() in image_extensions:
                records.append({
                    "image_id": image_file.stem,
                    "image_path": str(image_file),
                    "ethnic_group": ethnic_group,
                })

    df = pd.DataFrame(records)

    if len(df) == 0:
        logger.warning(f"No images found in {image_dir}")
        return df

    # Add label letters
    df["label_letter"] = df["ethnic_group"].map(ETHNIC_GROUP_TO_LABEL)

    # Sort for consistency
    df = df.sort_values(["ethnic_group", "image_id"]).reset_index(drop=True)

    if output_csv:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"Saved metadata to {output_csv}")

    logger.info(f"Created metadata for {len(df)} images")
    logger.info(f"Distribution: {df['ethnic_group'].value_counts().to_dict()}")

    return df


def split_by_language(
    metadata: pd.DataFrame,
    languages: List[str] = ["zh", "en"],
) -> pd.DataFrame:
    """Expand metadata for multiple prompt languages.

    Creates one row per image-language combination for Task 1.

    Args:
        metadata: DataFrame from load_metadata().
        languages: List of language codes.

    Returns:
        Expanded DataFrame with language column.
    """
    expanded_rows = []

    for _, row in metadata.iterrows():
        for lang in languages:
            new_row = row.to_dict()
            new_row["language"] = lang
            new_row["inference_id"] = f"{row['image_id']}_{lang}"
            expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)
    logger.info(
        f"Expanded {len(metadata)} images to {len(expanded_df)} "
        f"inference tasks ({len(languages)} languages)"
    )

    return expanded_df

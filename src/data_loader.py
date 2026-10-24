"""Dataset ingestion for the Edge-IIoTset ML dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def resolve_dataset_path(path=None) -> Path:
    """Return the dataset path, accepting either the plain or gzip-compressed CSV.

    ``data/ML-EdgeIIoT-dataset.csv`` is tried first, then
    ``data/ML-EdgeIIoT-dataset.csv.gz`` (the compressed copy that is committed
    to git, because GitHub rejects files larger than 100 MB).
    """
    path = Path(path or config.RAW_DATASET_PATH)
    candidates = [path]
    if path.suffix != ".gz":
        candidates.append(path.with_name(path.name + ".gz"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Dataset not found. Looked for: "
        + ", ".join(str(c) for c in candidates)
        + ". Place the Edge-IIoTset 'ML-EdgeIIoT-dataset.csv' (or its .csv.gz "
        "copy made by scripts/compress_dataset.py) under data/ first."
    )


def load_raw_dataset(path=None) -> pd.DataFrame:
    """Load the raw Edge-IIoTset ML CSV into a pandas DataFrame.

    Parameters
    ----------
    path : path-like, optional
        Override the default dataset location from ``config``. A ``.csv.gz``
        file is decompressed transparently by pandas.

    Returns
    -------
    pandas.DataFrame
        The raw dataset (63 columns).
    """
    resolved = resolve_dataset_path(path)
    df = pd.read_csv(resolved, low_memory=False)
    return df


def summarize_classes(df: pd.DataFrame, target: str = config.ATTACK_TYPE_COLUMN) -> pd.DataFrame:
    """Return per-class sample counts and proportions for the target column."""
    counts = df[target].value_counts()
    summary = pd.DataFrame(
        {
            "count": counts,
            "proportion": counts / counts.sum(),
        }
    )
    return summary

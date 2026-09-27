"""Data provenance tracking for Cerebro X.

Every processed dataset must have provenance. This module provides a simple
dataclass and JSON serializer for recording dataset identity and lineage.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class DatasetProvenance:
    """
    Records the identity and origin of a loaded dataset file.

    Fields:
        dataset_id:        Logical name (e.g. "oasis2_kaggle").
        file_path:         Absolute path to the source file.
        file_hash_md5:     MD5 hash of the source file for integrity verification.
        shape:             (rows, columns) of the loaded DataFrame.
        column_list:       Ordered list of column names.
        download_note:     Human-readable note about where/when data was obtained.
        processing_date:   ISO-8601 UTC timestamp when this provenance was created.
        software_note:     Optional note about software/library versions.
    """

    dataset_id: str
    file_path: str
    file_hash_md5: str
    shape: tuple[int, int]
    column_list: list[str]
    download_note: str
    processing_date: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    software_note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = list(d["shape"])  # make JSON-serializable
        return d


def compute_file_hash(path: str | Path) -> str:
    """
    Compute the MD5 hash of a file.

    Args:
        path: Path to the file.

    Returns:
        Lowercase hex MD5 digest string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot compute hash — file not found: {path}\n"
            f"Place oasis_longitudinal.csv in data/raw/ before running."
        )
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_provenance(
    dataset_id: str,
    file_path: str | Path,
    df,
    download_note: str,
    software_note: str = "",
) -> DatasetProvenance:
    """
    Build a DatasetProvenance from a loaded DataFrame and its source file.

    Args:
        dataset_id:     Logical dataset name.
        file_path:      Path to the CSV source file.
        df:             The loaded pandas DataFrame.
        download_note:  Free-text description of data origin.
        software_note:  Optional software versions string.

    Returns:
        DatasetProvenance instance.
    """
    file_path = Path(file_path).resolve()
    return DatasetProvenance(
        dataset_id=dataset_id,
        file_path=str(file_path),
        file_hash_md5=compute_file_hash(file_path),
        shape=tuple(df.shape),
        column_list=list(df.columns),
        download_note=download_note,
        software_note=software_note,
    )


def save_provenance(provenance: DatasetProvenance, output_path: str | Path) -> None:
    """
    Save provenance as a JSON file.

    Args:
        provenance:   DatasetProvenance instance.
        output_path:  Where to write the JSON (e.g. data/metadata/oasis2_provenance.json).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(provenance.to_dict(), fh, indent=2)


def load_provenance(path: str | Path) -> DatasetProvenance:
    """
    Load a previously saved DatasetProvenance from JSON.

    Args:
        path: Path to the JSON file.

    Returns:
        DatasetProvenance instance.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    data["shape"] = tuple(data["shape"])
    return DatasetProvenance(**data)

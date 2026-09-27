"""Shared IO utilities for Cerebro X."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict:
    """Load a YAML config file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def save_json(data: dict, path: str | Path, indent: int = 2) -> None:
    """Save a dictionary as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent)


def load_json(path: str | Path) -> dict:
    """Load a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def make_artifact_dir(base_dir: str | Path, experiment_id: str) -> Path:
    """
    Create and return the artifact directory for an experiment.

    Args:
        base_dir:      Root artifact directory (e.g. 'artifacts').
        experiment_id: Experiment ID string (e.g. 'EXP-BASELINE-001').

    Returns:
        Path to the created experiment artifact directory.
    """
    path = Path(base_dir) / experiment_id
    subdirs = ["model", "metrics", "plots", "config"]
    for subdir in subdirs:
        (path / subdir).mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def save_environment(output_path: str | Path) -> None:
    """
    Save pip freeze output to a file for reproducibility.

    Args:
        output_path: Where to write environment.txt.
    """
    import subprocess, sys
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=30,
        )
        env_text = f"Python: {sys.version}\n\n{result.stdout}"
        output_path.write_text(env_text, encoding="utf-8")
        logger.info("Environment saved: %s", output_path)
    except Exception as e:
        logger.warning("Could not save environment: %s", e)

#!/usr/bin/env python
"""
scripts/extract_eeg_features.py
--------------------------------
Phase 4: Extract spectral band-power features from ds004504 EEG recordings.

Reads EEGLAB .set files using MNE-Python, computes:
    - Delta (1-4 Hz)
    - Theta (4-8 Hz)
    - Alpha (8-13 Hz)
    - Beta  (13-30 Hz)
    - Gamma (30-45 Hz)

Per channel and as global averages across all 19 channels.

Output:
    data/processed/eeg_features.csv  — one row per subject, features + label

Label mapping:
    A (Alzheimer's)       → 0
    F (Frontotemporal)    → 1
    C (Control/Healthy)   → 2

Usage:
    python scripts/extract_eeg_features.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("extract_eeg_features")

EEG_DIR = Path("data/raw/eeg")
OUTPUT_PATH = Path("data/processed/eeg_features.csv")

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

GROUP_MAP = {"A": 0, "F": 1, "C": 2}
GROUP_NAMES = {0: "Alzheimer's", 1: "Frontotemporal Dementia", 2: "Control"}


def compute_band_powers(raw, sfreq: float) -> dict[str, float]:
    """Compute absolute band power per channel and global averages."""
    from scipy.signal import welch

    data = raw.get_data()  # (n_channels, n_times)
    n_ch = data.shape[0]
    features = {}

    all_band_powers = {band: [] for band in BANDS}

    for ch_idx in range(n_ch):
        ch_name = raw.ch_names[ch_idx]
        freqs, psd = welch(data[ch_idx], sfreq, nperseg=int(sfreq * 2))

        for band, (fmin, fmax) in BANDS.items():
            mask = (freqs >= fmin) & (freqs <= fmax)
            if mask.sum() > 0:
                power = np.trapz(psd[mask], freqs[mask])
            else:
                power = 0.0
            features[f"{ch_name}_{band}"] = float(power)
            all_band_powers[band].append(power)

    # Global averages across all channels
    for band, powers in all_band_powers.items():
        features[f"global_{band}"] = float(np.mean(powers))

    # Relative power (global)
    total_power = sum(features[f"global_{band}"] for band in BANDS) + 1e-10
    for band in BANDS:
        features[f"rel_global_{band}"] = features[f"global_{band}"] / total_power

    # Theta/Alpha ratio (useful AD biomarker)
    alpha = features.get("global_alpha", 1e-10)
    theta = features.get("global_theta", 0.0)
    features["theta_alpha_ratio"] = theta / (alpha + 1e-10)

    # Spectral edge frequency (95% power below this)
    try:
        freqs_all, psd_all = welch(data.mean(axis=0), sfreq, nperseg=int(sfreq * 2))
        cumulative = np.cumsum(psd_all) / (np.sum(psd_all) + 1e-10)
        sef95_idx = np.searchsorted(cumulative, 0.95)
        features["spectral_edge_freq_95"] = float(freqs_all[min(sef95_idx, len(freqs_all)-1)])
    except Exception:
        features["spectral_edge_freq_95"] = 0.0

    return features


def process_subject(sub_id: str, group_label: int, age: float, gender: str, mmse: float) -> dict | None:
    """Load EEG, preprocess, extract features for one subject."""
    try:
        import mne
        mne.set_log_level("WARNING")
    except ImportError:
        logger.error("MNE not installed. Run: pip install mne")
        return None

    set_path = EEG_DIR / sub_id / f"{sub_id}_task-eyesclosed_eeg.set"
    if not set_path.exists():
        logger.warning("EEG file not found: %s", set_path)
        return None

    try:
        raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    except Exception as e:
        logger.warning("Could not load %s: %s", sub_id, e)
        return None

    try:
        # Band-pass filter
        raw.filter(1.0, 45.0, fir_window="hamming", verbose=False)

        # Extract features
        features = compute_band_powers(raw, raw.info["sfreq"])

        # Metadata
        features["subject_id"] = sub_id
        features["group"] = group_label
        features["group_name"] = GROUP_NAMES[group_label]
        features["age"] = age
        features["gender"] = 1 if gender == "M" else 0
        features["mmse"] = mmse
        features["n_channels"] = len(raw.ch_names)
        features["sfreq"] = raw.info["sfreq"]
        features["duration_s"] = raw.times[-1]

        return features

    except Exception as e:
        logger.warning("Feature extraction failed for %s: %s", sub_id, e)
        return None


def main():
    if not EEG_DIR.exists():
        logger.error("EEG data directory not found: %s", EEG_DIR)
        logger.error("Run: python scripts/download_eeg_ds004504.py")
        sys.exit(1)

    participants_path = EEG_DIR / "participants.tsv"
    if not participants_path.exists():
        logger.error("participants.tsv not found. Run download script first.")
        sys.exit(1)

    # Load participant metadata
    participants = pd.read_csv(participants_path, sep="\t")
    logger.info("Participants: %d rows", len(participants))
    logger.info("Groups: %s", participants["Group"].value_counts().to_dict())

    all_features = []
    failed = []

    for _, row in participants.iterrows():
        sub_id = row["participant_id"]
        group_str = row["Group"]
        age = float(row.get("Age", 0))
        gender = str(row.get("Gender", "F"))
        mmse = float(row.get("MMSE", 0))

        if group_str not in GROUP_MAP:
            logger.warning("Unknown group '%s' for %s, skipping.", group_str, sub_id)
            continue

        group_label = GROUP_MAP[group_str]
        logger.info("Processing %s (Group=%s, Age=%s, MMSE=%s)...", sub_id, group_str, age, mmse)

        feat = process_subject(sub_id, group_label, age, gender, mmse)
        if feat is not None:
            all_features.append(feat)
        else:
            failed.append(sub_id)

    if not all_features:
        logger.error("No features extracted. Check EEG files in %s", EEG_DIR)
        sys.exit(1)

    features_df = pd.DataFrame(all_features)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_PATH, index=False)

    logger.info("=" * 60)
    logger.info("EEG FEATURE EXTRACTION COMPLETE")
    logger.info("Subjects processed: %d", len(all_features))
    logger.info("Subjects failed:    %d", len(failed))
    logger.info("Features per subject: %d", len(features_df.columns))
    logger.info("Output: %s", OUTPUT_PATH)
    logger.info("Label distribution: %s",
                features_df["group_name"].value_counts().to_dict())
    logger.info("=" * 60)

    if failed:
        logger.warning("Failed subjects: %s", failed)


if __name__ == "__main__":
    main()

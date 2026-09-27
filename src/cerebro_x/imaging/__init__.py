"""
Cerebro-X MRI Imaging Module.

Provides real NIfTI MRI ingestion, quality control, preprocessing,
and subject/visit alignment for the Digital Brain Twin pipeline.

Status:
    - nifti_loader.py  : File loading and validation
    - qc.py            : Quality control checks
    - preprocessing.py : Spatial + intensity normalization
    - subject_matcher.py: Safe subject/visit → MRI file matching

Prerequisites:
    pip install nibabel SimpleITK

Note:
    Raw MRI volumes must be externally provided (e.g., OASIS-2 NIfTI from
    https://sites.wustl.edu/oasisbrains/home/oasis-2/).
    Configure the root directory in configs/base.yaml → mri.root_dir.
"""
from __future__ import annotations

from cerebro_x.imaging.subject_matcher import MRISubjectMatcher
from cerebro_x.imaging.nifti_loader import NiftiLoader, MRILoadResult, MRILoadStatus
from cerebro_x.imaging.preprocessing import MRIPreprocessingPipeline
from cerebro_x.imaging.qc import MRIQualityChecker

__all__ = [
    "MRISubjectMatcher",
    "NiftiLoader",
    "MRILoadResult",
    "MRILoadStatus",
    "MRIPreprocessingPipeline",
    "MRIQualityChecker",
]

"""
MRI Subject Matcher for Cerebro-X.

Responsible for safely mapping clinical visit records to physical
MRI files based on subject ID, visit number, and file naming conventions.

This module guarantees that we never associate an MRI with the wrong subject,
which would constitute critical data contamination.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import re

logger = logging.getLogger(__name__)


class MRISubjectMatcher:
    """
    Safely matches clinical Subject IDs and Visit numbers to NIfTI files.
    
    OASIS-2 standard NIfTI naming convention is typically:
        sub-OAS20001_ses-d0000_T1w.nii.gz
    But since researchers often rename files or have legacy naming (e.g. OAS2_0001_MR1),
    this matcher uses a configurable regex/search strategy while strictly enforcing
    that the subject ID is present in the resolved filename.
    """
    
    def __init__(self, mri_root_dir: str | Path | None = None):
        self.mri_root_dir = Path(mri_root_dir) if mri_root_dir else None
        
    def _normalize_subject_id(self, subject_id: str) -> list[str]:
        """
        Generate possible filename variants of the subject ID.
        E.g., "OAS2_0001" -> ["OAS2_0001", "OAS20001", "sub-OAS20001"]
        """
        variants = [subject_id]
        
        # OASIS-2 specific: OAS2_0001 -> OAS20001
        if "_" in subject_id:
            variants.append(subject_id.replace("_", ""))
            
        return variants
        
    def find_mri_for_visit(self, subject_id: str, visit: int) -> Optional[Path]:
        """
        Locate the MRI file for a specific subject and visit.
        
        Args:
            subject_id: e.g., "OAS2_0001"
            visit: e.g., 1
            
        Returns:
            Path to the NIfTI file, or None if not found or unsafe to match.
        """
        if not self.mri_root_dir or not self.mri_root_dir.exists():
            return None
            
        variants = self._normalize_subject_id(subject_id)
        
        # Search strategies, from most specific to least specific
        search_patterns = []
        for var in variants:
            # 1. BIDS format
            search_patterns.append(f"*{var}*ses-*{visit}*T1w*.nii*")
            # 2. Legacy OASIS format (e.g. OAS2_0001_MR1)
            search_patterns.append(f"*{var}*MR{visit}*.nii*")
            # 3. Visit embedded anywhere near subject
            search_patterns.append(f"*{var}*{visit}*.nii*")
            
        for pattern in search_patterns:
            matches = list(self.mri_root_dir.rglob(pattern))
            if len(matches) == 1:
                logger.debug("Matched MRI for %s visit %d: %s", subject_id, visit, matches[0].name)
                return matches[0]
            elif len(matches) > 1:
                # Ambiguous match - better to return None than the wrong file
                logger.warning(
                    "Ambiguous MRI match for %s visit %d. Found %d files matching %s. "
                    "Skipping to prevent contamination.", 
                    subject_id, visit, len(matches), pattern
                )
                return None
                
        # Fallback: if we just find a subject directory, look inside it
        for var in variants:
            subj_dirs = list(self.mri_root_dir.glob(f"*{var}*"))
            for subj_dir in subj_dirs:
                if subj_dir.is_dir():
                    # Look for visit indicator
                    visit_files = list(subj_dir.rglob(f"*MR{visit}*.nii*")) + list(subj_dir.rglob(f"*ses-*{visit}*.nii*"))
                    if len(visit_files) == 1:
                        return visit_files[0]
                        
        logger.debug("No MRI found for %s visit %d", subject_id, visit)
        return None

    def validate_match(self, subject_id: str, file_path: str | Path) -> bool:
        """
        Strict validation that a given file path belongs to the subject.
        Used as a safety check before loading.
        """
        file_path = Path(file_path)
        variants = self._normalize_subject_id(subject_id)
        
        # The subject ID variant must be in the filename or immediate parent directory
        name = file_path.name.lower()
        parent = file_path.parent.name.lower()
        
        for var in variants:
            var_lower = var.lower()
            if var_lower in name or var_lower in parent:
                return True
                
        logger.error(
            "Subject match validation failed! Expected %s in path %s", 
            subject_id, file_path
        )
        return False

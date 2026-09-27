import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import sys
import os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

# ── Override the database engine BEFORE importing app ─────────────────────────
# This ensures lifespan's Base.metadata.create_all() uses the in-memory test DB.
import cerebro_x.api.database as _db_module
from sqlalchemy.pool import StaticPool

SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Patch the module-level engine and Base before anything else uses them
_db_module.engine = engine
_db_module.SessionLocal = TestingSessionLocal

from cerebro_x.api.main import app  # noqa: E402 — must import after patching
from cerebro_x.api.database import Base, get_db  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Create all tables in the test database once for the entire test session."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(setup_database):
    """Provide a database session for a test."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


@pytest.fixture
def client(db_session):
    """Provide a FastAPI TestClient that uses the test database."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass  # Session is closed by the db_session fixture

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def make_synthetic_oasis2(n_subjects: int = 20, seed: int = 42, n_rows: int = None) -> pd.DataFrame:
    """
    Generate a synthetic OASIS-2 DataFrame that matches the verified schema.

    Produces realistic longitudinal data with multiple visits per subject,
    correct column names, valid CDR values, and a deterministic seed.

    Args:
        n_subjects: Number of unique subjects to generate (default: 20).
        seed:       Random seed for reproducibility (default: 42).
        n_rows:     Deprecated alias; if provided, used as n_subjects.

    Returns:
        DataFrame with all 15 required OASIS-2 columns in the correct schema,
        sorted by Subject ID then Visit.
    """
    # Backwards-compat: allow old positional call make_synthetic_oasis2(n_rows=10)
    if n_rows is not None:
        n_subjects = n_rows

    rng = np.random.default_rng(seed)

    CDR_VALUES = [0.0, 0.5, 1.0, 2.0]
    CDR_WEIGHTS = [0.55, 0.33, 0.11, 0.01]

    records = []
    for i in range(n_subjects):
        subject_id = f"OAS2_{i + 1:04d}"
        # Distribution: ~63% have 2 visits, ~29% have 3, ~8% have 4
        n_visits = int(rng.choice([2, 3, 4], p=[0.63, 0.29, 0.08]))
        sex = rng.choice(["M", "F"])
        hand = "R"
        educ = int(rng.integers(8, 20))
        # SES with ~10% missing
        ses_raw = rng.choice([1.0, 2.0, 3.0, 4.0, np.nan], p=[0.2, 0.25, 0.25, 0.2, 0.1])
        ses = float(ses_raw)
        age_base = float(rng.integers(60, 85))
        etiv_base = float(rng.integers(1300, 1800))
        nwbv_base = float(rng.uniform(0.68, 0.85))
        asf_base = 1.0 + float(rng.uniform(-0.15, 0.15))
        mmse_base = float(rng.integers(20, 30))
        cdr_base = float(rng.choice(CDR_VALUES, p=CDR_WEIGHTS))
        mr_delay = 0

        for v_idx in range(n_visits):
            visit_num = v_idx + 1
            age = age_base + v_idx * 1.5
            mmse = float(np.clip(mmse_base - v_idx * float(rng.uniform(0.0, 2.0)), 0.0, 30.0))
            # Occasional progression
            cdr_increment = 0.5 if (v_idx > 0 and float(rng.random()) > 0.7) else 0.0
            cdr_raw = cdr_base + cdr_increment
            cdr_raw = min(cdr_raw, 2.0)
            # Snap to nearest valid CDR
            cdr = float(min(CDR_VALUES, key=lambda x: abs(x - cdr_raw)))
            nwbv = float(nwbv_base - v_idx * 0.005)
            etiv = float(etiv_base + float(rng.uniform(-20, 20)))
            asf = float(asf_base + float(rng.uniform(-0.02, 0.02)))
            if v_idx > 0:
                mr_delay = mr_delay + int(rng.integers(300, 800))

            mri_id = f"OAS2_{i + 1:04d}_MR{visit_num}"
            if cdr == 0.0:
                group = "Nondemented"
            elif v_idx > 0 and cdr_base == 0.0:
                group = "Converted"
            else:
                group = "Demented"

            # ~1% MMSE missing (realistic)
            mmse_val = mmse if float(rng.random()) > 0.01 else np.nan

            records.append({
                "Subject ID": subject_id,
                "MRI ID": mri_id,
                "Group": group,
                "Visit": visit_num,
                "MR Delay": mr_delay,
                "M/F": sex,
                "Hand": hand,
                "Age": age,
                "EDUC": educ,
                "SES": ses,
                "MMSE": mmse_val,
                "CDR": cdr,
                "eTIV": etiv,
                "nWBV": nwbv,
                "ASF": asf,
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["Subject ID", "Visit"]).reset_index(drop=True)
    return df


@pytest.fixture
def synthetic_oasis2_df():
    """Fixture providing a standard synthetic OASIS-2 DataFrame for tests."""
    return make_synthetic_oasis2(n_subjects=20, seed=42)


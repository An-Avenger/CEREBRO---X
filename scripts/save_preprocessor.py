"""Script to save the fitted ClinicalPreprocessor from the existing training data.
Run once after cloning repo to regenerate preprocessor artifact."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cerebro_x.data.pytorch.sequence_dataset import SequenceDataset
from cerebro_x.features.preprocessor import ClinicalPreprocessor
from cerebro_x.evaluation.splits import subject_level_split
import pandas as pd

pairs_path = Path('data/processed/next_visit_pairs.csv')
if not pairs_path.exists():
    print(f'ERROR: pairs CSV not found at {pairs_path}. Run scripts/build_pairs.py first.')
    sys.exit(1)

pairs_df = pd.read_csv(pairs_path)
print(f'Loaded {len(pairs_df)} pairs from {pairs_df["Subject ID"].nunique()} subjects')

train_df, val_df, test_df, _ = subject_level_split(
    pairs_df, test_size=0.15, val_size=0.15, seed=42
)

train_ds = SequenceDataset(train_df, is_train=True)
print(f'Train dataset: {len(train_ds)} samples, {train_ds.num_features} features')

prep = ClinicalPreprocessor()
prep.imputer = train_ds.imputer
prep.scaler = train_ds.scaler
prep.is_fitted = True

artifact_dir = Path('artifacts/EXP-LONGITUDINAL-001')
prep.save(artifact_dir)
print(f'ClinicalPreprocessor saved to {artifact_dir}')
print(f'Feature order ({len(prep.feature_order)}): {prep.feature_order}')

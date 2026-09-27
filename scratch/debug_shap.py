import sys
from pathlib import Path
sys.path.insert(0, str(Path('src')))
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from cerebro_x.data.pytorch.sequence_dataset import SequenceDataset, sequence_collate_fn
from cerebro_x.evaluation.splits import subject_level_split
from cerebro_x.models.deep.temporal import TemporalCerebroNet
from cerebro_x.explainability.shap_temporal import compute_temporal_shap
import torch.nn.functional as F

pairs_df = pd.read_csv('data/processed/next_visit_pairs.csv')
train_df, _, test_df, _ = subject_level_split(pairs_df, test_size=0.15, val_size=0.15, seed=42, target_col='next_cdr')
train_ds = SequenceDataset(train_df, is_train=True)
test_ds = SequenceDataset(test_df, imputer=train_ds.imputer, scaler=train_ds.scaler, is_train=False)

model = TemporalCerebroNet(input_dim=train_ds.num_features, gru_hidden_dim=64, gru_num_layers=1, head_hidden_dim=32, dropout=0.3)
model.load_state_dict(torch.load('artifacts/EXP-LONGITUDINAL-001/temporal_gru_cpu.pt', map_location='cpu'))
model.eval()

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=sequence_collate_fn)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=sequence_collate_fn)
bg_features, _, bg_lengths = next(iter(train_loader))
test_features, _, test_lengths = next(iter(test_loader))
max_seq_len = max(bg_features.size(1), test_features.size(1))
bg_features = F.pad(bg_features, (0, 0, 0, max_seq_len - bg_features.size(1)))
test_features = F.pad(test_features, (0, 0, 0, max_seq_len - test_features.size(1)))

shap_values_raw = compute_temporal_shap(model, bg_features, bg_lengths, test_features, test_lengths)
print("SHAP VALUES RAW TYPE:", type(shap_values_raw))
if isinstance(shap_values_raw, list):
    print("List length:", len(shap_values_raw))
    print("Element shape:", shap_values_raw[0].shape)
else:
    print("Raw shape:", shap_values_raw.shape)

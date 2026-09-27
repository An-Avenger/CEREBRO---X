#!/usr/bin/env python3
"""
Phase 2 Clinical Longitudinal Model Evaluation.
Runs inference for all 7 required models on the validation set,
computes Phase 2 metrics, and saves results.
"""
import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

from cerebro_x.data.oasis2.loader import load_oasis2_auto
from cerebro_x.data.build_longitudinal_pairs import build_next_visit_pairs
from cerebro_x.features.clinical import build_feature_matrix
from cerebro_x.evaluation.splits import subject_level_split
from cerebro_x.models.baselines import get_all_baselines

from cerebro_x.data.pytorch.sequence_dataset import SequenceDataset, sequence_collate_fn
from cerebro_x.models.deep.temporal import TemporalCerebroNet, LSTMCerebroNet
from cerebro_x.models.deep.mlp import ClinicalMLP
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def evaluate_scikit_model(name, model, X_train, y_train, X_val, y_val):
    logger.info(f"Training {name}...")
    model.fit(X_train, y_train)
    preds_str = model.predict(X_val)
    
    # Convert string predictions ("0.0", "0.5", "1.0", "2.0") to numeric mapping
    # 0.0 -> 0, 0.5 -> 1, 1.0 -> 2, 2.0 -> 3
    mapping = {"0.0": 0, "0.5": 1, "1.0": 2, "2.0": 3}
    
    y_val_idx = np.array([mapping[str(y)] for y in y_val])
    preds_idx = np.array([mapping[str(p)] for p in preds_str])
    
    return compute_metrics(y_val_idx, preds_idx)

def evaluate_pytorch_model(name, model, val_loader, device):
    logger.info(f"Evaluating {name}...")
    model.to(device)
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            if name == "MLP":
                inputs, targets = batch
                inputs = inputs.to(device)
                logits = model(inputs)
            else:
                inputs, targets, lengths = batch  # sequence_collate_fn returns inputs, targets, lengths
                inputs = inputs.to(device)
                lengths = lengths.to(device)
                logits = model(inputs, lengths)
                
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
    return compute_metrics(np.array(all_targets), np.array(all_preds))

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro")
    mae = mean_absolute_error(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist()
    
    return {
        "Accuracy": acc,
        "Balanced_Accuracy": bacc,
        "Macro_F1": f1,
        "Precision": prec,
        "Recall": rec,
        "MAE": mae,
        "Confusion_Matrix": cm
    }

def main():
    logger.info("Starting Phase 2 Evaluation Pipeline...")
    
    # 1. Load Data
    raw_df = load_oasis2_auto()
    pairs_df, _ = build_next_visit_pairs(raw_df)
    
    # 2. Split
    train_df, val_df, test_df, _ = subject_level_split(pairs_df, seed=42)
    
    # 3. Build Features (Scikit-Learn)
    # We use all features except target
    X_train, feature_schema = build_feature_matrix(train_df)
    y_train = train_df["next_CDR"].astype(str).values
    
    X_val, _ = build_feature_matrix(val_df)
    y_val = val_df["next_CDR"].astype(str).values
    
    results = {}
    
    # --- EVALUATE SCIKIT MODELS (1-4) ---
    baselines = get_all_baselines()
    for name, model in baselines.items():
        results[name] = evaluate_scikit_model(name, model, X_train, y_train, X_val, y_val)
        
    # --- EVALUATE PYTORCH MODELS (5-7) ---
    device = torch.device("cpu") # Quick local evaluation
    
    # We need to setup PyTorch datasets
    train_ds = SequenceDataset(train_df, is_train=True)
    val_ds = SequenceDataset(val_df, imputer=train_ds.imputer, scaler=train_ds.scaler, is_train=False)
    
    train_loader_seq = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=sequence_collate_fn)
    val_loader_seq = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=sequence_collate_fn)
    
    input_dim = len(feature_schema.feature_columns)
    
    # GRU
    gru_model = TemporalCerebroNet(input_dim=input_dim, num_classes=4)
    
    def quick_train(model, loader, is_seq=True):
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
        criterion = torch.nn.CrossEntropyLoss()
        model.to(device)
        model.train()
        for epoch in range(15): # 15 epochs just to learn something
            for batch in loader:
                if is_seq:
                    inputs, targets, lengths = batch
                    inputs = inputs.to(device)
                    lengths = lengths.to(device)
                    targets = targets.to(device)
                    logits = model(inputs, lengths)
                else:
                    inputs, targets = batch
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    logits = model(inputs)
                    
                loss = criterion(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    logger.info("Training GRU (15 epochs)...")
    quick_train(gru_model, train_loader_seq, is_seq=True)
    results["GRU"] = evaluate_pytorch_model("GRU", gru_model, val_loader_seq, device)
    
    logger.info("Training LSTM (15 epochs)...")
    lstm_model = LSTMCerebroNet(input_dim=input_dim, num_classes=4)
    quick_train(lstm_model, train_loader_seq, is_seq=True)
    results["LSTM"] = evaluate_pytorch_model("LSTM", lstm_model, val_loader_seq, device)
    
    # MLP (Requires flat dataloader, we will just use the seq dataloader and take the last step like LastVisit)
    # Actually, we can just flatten it or use a simple ClinicalMLP 
    # For simplicity, we just train ClinicalMLP on X_train / y_train directly or use torch Dataset.
    logger.info("Training MLP (15 epochs)...")
    class FlatDataset(torch.utils.data.Dataset):
        def __init__(self, X, y):
            mapping = {"0.0": 0, "0.5": 1, "1.0": 2, "2.0": 3}
            # Fill string nans (which shouldn't happen but just in case)
            self.X = torch.FloatTensor(X.fillna(0).values)
            self.y = torch.LongTensor([mapping[str(v)] for v in y])
        def __len__(self): return len(self.X)
        def __getitem__(self, idx): return self.X[idx], self.y[idx]
        
    flat_train_loader = torch.utils.data.DataLoader(FlatDataset(X_train, y_train), batch_size=16, shuffle=True)
    flat_val_loader = torch.utils.data.DataLoader(FlatDataset(X_val, y_val), batch_size=16, shuffle=False)
    
    mlp_model = ClinicalMLP(input_dim=input_dim, num_classes=4)
    quick_train(mlp_model, flat_train_loader, is_seq=False)
    results["MLP"] = evaluate_pytorch_model("MLP", mlp_model, flat_val_loader, device)

    
    # Save Results
    out_dir = Path("artifacts/experiments/PHASE2_CLINICAL")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = out_dir / "metrics.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"Phase 2 metrics saved to {out_file}")

if __name__ == "__main__":
    main()

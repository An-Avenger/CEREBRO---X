import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path("src")))

from cerebro_x.models.deep.temporal import TemporalCerebroNet
from cerebro_x.explainability.shap_temporal import compute_temporal_shap

model = TemporalCerebroNet(input_dim=19, gru_hidden_dim=64, num_classes=4)
state = torch.load("artifacts/EXP-LONGITUDINAL-001/shap_background.pt", map_location="cpu", weights_only=True)
bg = state["background"][:2]
bg_len = state["lengths"][:2]
test_x = bg[:1]
test_len = bg_len[:1]

shap_values = compute_temporal_shap(model, bg, bg_len, test_x, test_len)

print("TYPE:", type(shap_values))
if isinstance(shap_values, list):
    print("LIST LEN:", len(shap_values))
    print("ITEM TYPE:", type(shap_values[0]))
    if hasattr(shap_values[0], "shape"):
        print("ITEM SHAPE:", shap_values[0].shape)
elif hasattr(shap_values, "shape"):
    print("SHAPE:", shap_values.shape)

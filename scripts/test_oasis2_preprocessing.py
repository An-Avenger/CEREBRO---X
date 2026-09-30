import json
from pathlib import Path
import torch
from cerebro_x.imaging.oasis2_dataset import RealOASIS2MRIDataset
from torch.utils.data import DataLoader
import psutil
import os

def run_preprocessing_test():
    manifest_path = Path("D:/CEREBRO_DATA/OASIS2/manifests/oasis2_mri_clinical_aligned.csv")
    if not manifest_path.exists():
        print("Manifest not found! Skipping preprocessing test.")
        return
        
    print("Running Preprocessing Test on real OASIS-2 MRI data...")
    dataset = RealOASIS2MRIDataset(manifest_path=manifest_path, augment=False, target_shape=(64, 64, 64))
    
    test_size = min(5, len(dataset))
    print(f"Testing first {test_size} volumes...")
    
    report = {
        "success": True,
        "samples_tested": test_size,
        "results": []
    }
    
    try:
        for i in range(test_size):
            print(f"Loading sample {i+1}...")
            vol_tensor, label, subject_id = dataset[i]
            
            # Checks
            has_nan = torch.isnan(vol_tensor).any().item()
            has_inf = torch.isinf(vol_tensor).any().item()
            
            mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) # MB
            
            result = {
                "subject_id": subject_id,
                "label": int(label),
                "tensor_shape": list(vol_tensor.shape),
                "has_nan": has_nan,
                "has_inf": has_inf,
                "min_val": float(vol_tensor.min()),
                "max_val": float(vol_tensor.max()),
                "memory_mb": round(mem, 2)
            }
            report["results"].append(result)
            
            if has_nan or has_inf or list(vol_tensor.shape) != [1, 64, 64, 64]:
                report["success"] = False
                print(f"FAILED on sample {i+1}: {result}")
                
        print("Preprocessing test completed.")
    except Exception as e:
        report["success"] = False
        report["error"] = str(e)
        print(f"FAILED: {e}")
        
    report_path = Path("D:/CEREBRO_DATA/OASIS2/logs/preprocessing_test_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    run_preprocessing_test()

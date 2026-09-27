import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def run_tests():
    print("=== 1. Health Check ===")
    try:
        res = requests.get(f"{BASE_URL}/health")
        print("Status:", res.status_code, res.json())
    except Exception as e:
        print("Error:", e)

    cases = {
        "CASE A (Healthy)": {
            "subject_id": "DEMO_A",
            "visits": [{"age": 70, "educ": 16, "ses": 2, "mmse": 29, "cdr": 0.0, "nwbv": 0.78, "etiv": 1500, "asf": 1.10, "apoe4": False, "p_tau": None}],
            "mri": {"nwbv": 0.78, "etiv": 1500, "asf": 1.10, "nwbv_delta": 0.0}
        },
        "CASE B (Impaired)": {
            "subject_id": "DEMO_B",
            "visits": [{"age": 74, "educ": 16, "ses": 2, "mmse": 22, "cdr": 1.0, "nwbv": 0.72, "etiv": 1500, "asf": 1.10, "apoe4": False, "p_tau": None}],
            "mri": {"nwbv": 0.72, "etiv": 1500, "asf": 1.10, "nwbv_delta": 0.0}
        },
        "CASE C (Intermediate)": {
            "subject_id": "DEMO_C",
            "visits": [{"age": 72, "educ": 16, "ses": 2, "mmse": 26, "cdr": 0.5, "nwbv": 0.75, "etiv": 1500, "asf": 1.10, "apoe4": False, "p_tau": None}],
            "mri": {"nwbv": 0.75, "etiv": 1500, "asf": 1.10, "nwbv_delta": 0.0}
        }
    }

    for name, payload in cases.items():
        print(f"\n=== {name} ===")
        res = requests.post(f"{BASE_URL}/predict/bimodal", json=payload)
        if res.status_code == 200:
            print("Probabilities:", json.dumps(res.json().get("class_probabilities", {}), indent=2))
            print("Verdict:", res.json().get("predicted_cdr_label"))
        else:
            print(f"Error {res.status_code}: {res.text}")

    print("\n=== Multi-Visit Brain Twin Extraction ===")
    history_payload = {
        "subject_id": "DEMO_HISTORY",
        "visits": [
            {"age": 70, "educ": 16, "ses": 2, "mmse": 29, "cdr": 0.0, "nwbv": 0.78, "etiv": 1500, "asf": 1.10, "apoe4": False, "p_tau": None},
            {"age": 72, "educ": 16, "ses": 2, "mmse": 27, "cdr": 0.5, "nwbv": 0.76, "etiv": 1500, "asf": 1.10, "apoe4": False, "p_tau": None},
            {"age": 74, "educ": 16, "ses": 2, "mmse": 22, "cdr": 1.0, "nwbv": 0.72, "etiv": 1500, "asf": 1.10, "apoe4": False, "p_tau": None}
        ]
    }
    res = requests.post(f"{BASE_URL}/brain-twin/extract", json=history_payload)
    if res.status_code == 200:
        data = res.json()
        print("Trajectory length:", len(data.get("trajectory", [])))
        print("Latest BHI:", data.get("bhi_scores", [])[-1] if data.get("bhi_scores") else "N/A")
    else:
        print(f"Error {res.status_code}: {res.text}")

    print("\n=== History Verification ===")
    res = requests.get(f"{BASE_URL}/history/")
    if res.status_code == 200:
        history = res.json()
        print("Total history records:", len(history))
    else:
        print(f"Error {res.status_code}: {res.text}")
        
    print("\n=== Experiments Verification ===")
    res = requests.get(f"{BASE_URL}/experiments/")
    if res.status_code == 200:
        exps = res.json()
        print("Total experiments found:", len(exps))
    else:
        print(f"Error {res.status_code}: {res.text}")

if __name__ == "__main__":
    run_tests()

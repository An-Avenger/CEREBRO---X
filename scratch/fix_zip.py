import os
import zipfile

def create_kaggle_zip():
    zip_path = "cerebro_x_codebase.zip"
    dirs_to_zip = ["src", "scripts", "configs", "data/processed", "tests", "docs"]
    files_to_zip = ["pyproject.toml", "requirements.txt"]

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in files_to_zip:
            if os.path.exists(f):
                zf.write(f, f)
                
        for d in dirs_to_zip:
            # normalize for windows just in case
            d = d.replace("/", os.sep)
            if os.path.exists(d):
                for root, _, files in os.walk(d):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # CRITICAL: Kaggle requires forward slashes (/) for paths inside the zip
                        arcname = file_path.replace(os.sep, '/')
                        zf.write(file_path, arcname)
                        print(f"Added {arcname}")

create_kaggle_zip()
print("ZIP RECREATED SUCCESSFULLY WITH FORWARD SLASHES.")

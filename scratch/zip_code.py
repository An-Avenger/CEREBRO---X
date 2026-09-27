import os
import zipfile

def create_zip():
    zip_path = "cerebro_x_code.zip"
    directories_to_zip = ["src", "scripts"]
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for directory in directories_to_zip:
            for root, _, files in os.walk(directory):
                for file in files:
                    # Ignore pycache
                    if "__pycache__" in root or file.endswith(".pyc"):
                        continue
                        
                    file_path = os.path.join(root, file)
                    
                    # Create archive name with FORWARD SLASHES for Kaggle compatibility
                    arcname = os.path.relpath(file_path, start=".")
                    arcname = arcname.replace(os.sep, '/')
                    
                    zipf.write(file_path, arcname)
                    print(f"Added: {arcname}")
                    
    print(f"\nSuccessfully created {zip_path} with forward slashes!")

if __name__ == "__main__":
    create_zip()

import re
from pathlib import Path

file_path = Path(r"e:\PROJECTS\CEREBRO-X\tests\test_explainability.py")
content = file_path.read_text(encoding="utf-8")

# Remove singleton reset and client overwrite in test_mri_explain_with_loaded_model
content = re.sub(
    r'        # Reset singleton so it can reload with CNN3D\n.*?client = TestClient\(app\)\n',
    '',
    content,
    flags=re.DOTALL
)

# Remove client overwrite in test_mri_explain_response_has_correct_method_when_successful
content = re.sub(
    r'        client = TestClient\(app\)\n',
    '',
    content
)

file_path.write_text(content, encoding="utf-8")
print("test_explainability.py patched again")

import re
from pathlib import Path

file_path = Path(r"e:\PROJECTS\CEREBRO-X\tests\test_explainability.py")
content = file_path.read_text(encoding="utf-8")

# Replace all occurrences of client = self._get_client() with nothing
content = re.sub(r'^[ \t]*client = self\._get_client\(\)[ \t]*\n', '', content, flags=re.MULTILINE)

# Ensure all test methods that belong to the API classes take `client` as an argument
# We know the classes are TestAPIClinicalSHAP, TestAPIMRIGradCAM, TestExistingFunctionalityRegression
# Any method starting with `def test_` in these classes should have `client`
# A simpler way: just replace `def test_(\w+)\(self\):` with `def test_\1(self, client):` for ALL tests that don't already have it
content = re.sub(r'(def test_\w+)\(self\):', r'\1(self, client):', content)

file_path.write_text(content, encoding="utf-8")
print("test_explainability.py patched robustly")

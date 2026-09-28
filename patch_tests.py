import re
from pathlib import Path

file_path = Path(r"e:\PROJECTS\CEREBRO-X\tests\test_explainability.py")
content = file_path.read_text(encoding="utf-8")

# Remove all _get_client methods entirely
content = re.sub(
    r'    def _get_client\(self\):\n(?:        [^\n]+\n)+',
    '',
    content
)

# For every test that calls client = self._get_client(), inject client parameter
def replacer(match):
    # Match group 1: method definition up to self
    # Match group 2: method body
    head = match.group(1)
    body = match.group(2)
    # add client to args if not there
    if not head.endswith(", client)"):
        head = head.replace("(self):", "(self, client):")
    # remove client = self._get_client() from body
    body = re.sub(r' +client = self\._get_client\(\)\n', '', body)
    return head + body

content = re.sub(r'(    def test_\w+\(self\):)(\n(?:        [^\n]*\n)*?)', replacer, content)

file_path.write_text(content, encoding="utf-8")
print("test_explainability.py patched successfully")

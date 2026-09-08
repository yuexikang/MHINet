from pathlib import Path
import hashlib
import json
import sys

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
errors = []
for relative, expected in manifest.items():
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        errors.append('Path outside package: ' + relative)
    elif not path.is_file():
        errors.append('Missing: ' + relative)
    elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        errors.append('Hash mismatch: ' + relative)
print(json.dumps(dict(files_checked=len(manifest), errors=errors), indent=2))
sys.exit(1 if errors else 0)

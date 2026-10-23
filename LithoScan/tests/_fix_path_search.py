"""Fix the test: search only the filename (Path(fp).name), not the full path."""
from pathlib import Path

target = Path("tests/test_dataset.py")
content = target.read_text(encoding="utf-8")

# Fix the generator expression to check only the filename
old = 'next(\n            i for i, (fp, _) in enumerate(ds.samples) if "corrupt" in fp\n        )\n        import pytest\n        with pytest.raises(RuntimeError):\n            _ = ds[corrupt_idx]'

new = '# Use Path(fp).name to match only the filename, NOT the full path.\n        # (pytest tmp_path dirs contain the test name which includes "corrupt".)\n        next(\n            i for i, (fp, _) in enumerate(ds.samples) if "corrupt" in Path(fp).name\n        )\n        import pytest\n        with pytest.raises(RuntimeError):\n            _ = ds[corrupt_idx]'

if old in content:
    content = content.replace(old, new, 1)
    target.write_text(content, encoding="utf-8")
    print("Fixed: now matching on Path(fp).name only.")
else:
    print("Pattern not found. Dumping context around 'corrupt in fp'...")
    idx = content.find('"corrupt" in fp')
    if idx != -1:
        print(repr(content[max(0, idx-100):idx+120]))

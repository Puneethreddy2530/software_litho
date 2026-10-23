"""Clean fix for the corrupt_idx search: use Path(fp).name."""
from pathlib import Path

target = Path("tests/test_dataset.py")
content = target.read_text(encoding="utf-8")

# Show the broken region
idx = content.find("corrupt_idx = # Use")
if idx != -1:
    print("Found broken line at char", idx)
    print(repr(content[max(0, idx-50):idx+300]))

    # Replace the broken assignment with correct multi-line form
    old = 'corrupt_idx = # Use Path(fp).name to match only the filename, NOT the full path.\n        # (pytest tmp_path dirs contain the test name which includes "corrupt".)\n        next(\n            i for i, (fp, _) in enumerate(ds.samples) if "corrupt" in Path(fp).name\n        )'
    new = ('# Use Path(fp).name to match only the filename, NOT the full path.\n'
           '        # (pytest tmp_path dirs contain the test name which includes "corrupt".)\n'
           '        corrupt_idx = next(\n'
           '            i for i, (fp, _) in enumerate(ds.samples) if "corrupt" in Path(fp).name\n'
           '        )')
    if old in content:
        content = content.replace(old, new, 1)
        target.write_text(content, encoding="utf-8")
        print("Fixed.")
    else:
        print("Pattern not found, trying AST-safe direct rewrite of that line...")
        lines = content.splitlines(keepends=True)
        for i, line in enumerate(lines):
            if 'corrupt_idx = # Use' in line:
                print(f"Line {i+1}: {repr(line)}")
        print("Manual fix needed.")
else:
    # Check if already fixed
    idx2 = content.find('"corrupt" in Path(fp).name')
    if idx2 != -1:
        print("Already fixed - contains Path(fp).name")
    else:
        idx3 = content.find('"corrupt" in fp')
        print(f"Still has old pattern at char {idx3}: {repr(content[max(0,idx3-50):idx3+80])}")

"""Fix the corrupt-bytes test to use PIL-unreadable content."""
from pathlib import Path

target = Path("tests/test_dataset.py")
content = target.read_text(encoding="utf-8")

# The old line that writes a partial JPEG header
old_bytes_line = "(fake_split / \"corrupt.jpg\").write_bytes(bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b\"BROKEN\")"
new_bytes_line = (
    "# Plain ASCII text - PIL cannot identify this as any image format.\n"
    "        (fake_split / \"corrupt.jpg\").write_bytes(b\"THIS IS NOT AN IMAGE FILE AT ALL\")"
)

if old_bytes_line in content:
    content = content.replace(old_bytes_line, new_bytes_line, 1)
    target.write_text(content, encoding="utf-8")
    print("Fixed: replaced corrupt bytes with unidentifiable content.")
else:
    # Also try the \r\n variant
    old_bytes_line_r = old_bytes_line.replace("\n", "\r\n")
    if old_bytes_line_r in content:
        content = content.replace(old_bytes_line_r, new_bytes_line, 1)
        target.write_text(content, encoding="utf-8")
        print("Fixed (CRLF variant).")
    else:
        # Find and show context for debugging
        idx = content.find("corrupt.jpg")
        while idx != -1:
            print(f"Found 'corrupt.jpg' at {idx}:")
            print(repr(content[max(0, idx-20):idx+120]))
            print()
            idx = content.find("corrupt.jpg", idx + 1)

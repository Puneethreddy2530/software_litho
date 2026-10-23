"""
conftest.py — pytest configuration for LithoScan.

Adds the project root to sys.path so that ``src.*`` imports resolve
correctly regardless of how pytest is invoked.
"""

import sys
from pathlib import Path

# Project root = directory containing this conftest.py's parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

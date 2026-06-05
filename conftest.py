"""Pytest setup: make the app + harness importable and isolate run output to a
temp DATA_DIR so tests never touch a real results disk."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))

# api.py reads DATA_DIR at import time — point it at a throwaway dir for tests.
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="fcb_test_"))

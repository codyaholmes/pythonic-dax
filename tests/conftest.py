from __future__ import annotations

import sys
from pathlib import Path

# Make `golden.harness` importable from tests.
sys.path.insert(0, str(Path(__file__).parent))

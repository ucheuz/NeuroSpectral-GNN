#!/usr/bin/env python
"""Wrapper: BIDS (or similar) NIfTI + PRS pre-flight. See ``src.preprocessing.bids_validator``."""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.preprocessing.bids_validator import main

if __name__ == "__main__":
    raise SystemExit(main())

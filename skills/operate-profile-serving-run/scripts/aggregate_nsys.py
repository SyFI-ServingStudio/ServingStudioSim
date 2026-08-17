#!/usr/bin/env python3
"""Compatibility entrypoint for the shared alignment NSYS evidence extractor."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def main() -> int:
    """Delegate to the repository-owned evidence extractor."""
    from alignment.nsys.evidence import main as evidence_main

    return evidence_main()


if __name__ == "__main__":
    raise SystemExit(main())

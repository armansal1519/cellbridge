#!/usr/bin/env python3
"""Check the frozen solver against the GSE334503 freeze."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from responsebridge.emtab_adapter import assert_gse_frozen_sources  # noqa: E402


def main():
    found = assert_gse_frozen_sources(ROOT)
    for rel, digest in found.items():
        print(f"{rel} {digest}")
    print("frozen solver matches the GSE334503 freeze")


if __name__ == "__main__":
    main()

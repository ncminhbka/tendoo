#!/usr/bin/env python3
"""
CLI wrapper for Tendoo AI Production Glyph Engine.
Usage:
    python scripts/generate_glyph.py --inspect
    python scripts/generate_glyph.py --test_suite
    python scripts/generate_glyph.py --text "CÀ PHÊ SỮA ĐÁ" --font "bevietnam" --output "glyph.png"
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tendoo.glyph_engine import main

if __name__ == "__main__":
    main()

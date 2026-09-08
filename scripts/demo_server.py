#!/usr/bin/env python3
"""
scripts/demo_server.py
Forwarding launcher for tendoo.demo_server.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in [PROJECT_ROOT, PROJECT_ROOT / "src"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tendoo.demo_server import *
from tendoo.demo_server import main

if __name__ == "__main__":
    main()

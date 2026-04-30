#!/usr/bin/env python3
"""CLI script for generating expert evaluation materials.

Usage:
    python scripts/generate_human_eval.py --results-dir results/raw --output-dir results/human_eval
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.human_eval.generate_sheets import main

if __name__ == "__main__":
    sys.exit(main())

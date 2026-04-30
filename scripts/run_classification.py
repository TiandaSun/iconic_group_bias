#!/usr/bin/env python3
"""CLI script for running Task 1: Classification.

Usage:
    python scripts/run_classification.py --model qwen2.5-vl-7b --language zh --data data/metadata.csv
    python scripts/run_classification.py -m gpt-4o-mini -l en -d data/metadata.csv -o results/raw
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.task1_classification import main

if __name__ == "__main__":
    sys.exit(main())

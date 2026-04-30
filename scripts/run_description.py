#!/usr/bin/env python3
"""CLI script for running Task 2: Description Generation.

Usage:
    python scripts/run_description.py --model qwen2.5-vl-7b --language zh --data data/task2_samples.csv
    python scripts/run_description.py -m gpt-4o-mini -l en -d data/metadata.csv -n 500

Total inference: 7 models x 2 languages x 500 images = 7,000 descriptions
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inference.task2_description import main

if __name__ == "__main__":
    sys.exit(main())

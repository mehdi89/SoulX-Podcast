#!/usr/bin/env python3
"""
Entry point for the SoulX-Podcast worker.

Usage:
    python run_worker.py
    python run_worker.py --env /path/to/.env
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from worker.main import run_worker


def main():
    parser = argparse.ArgumentParser(
        description="SoulX-Podcast Worker - Processes podcast generation jobs"
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Path to .env file (default: worker/.env or .env)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  SoulX-Podcast Worker")
    print("=" * 60)
    print()

    run_worker(env_path=args.env)


if __name__ == "__main__":
    main()

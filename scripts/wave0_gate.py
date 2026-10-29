"""Print the Wave 0 gate table for the M1 gate report.

Lane: Hemdutt Rao (market layer).

    python -m scripts.wave0_gate --episodes 200
"""

from __future__ import annotations

import argparse
import sys

from optspread.gates import wave0_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    report = wave0_gate(args.episodes, args.seed)
    print(report.render())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())

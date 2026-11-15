"""Print the Wave 0 gate table for the M1 gate report.

Lane: Hemdutt Rao (market layer).

    python -m scripts.wave0_gate --episodes 200
    python -m scripts.wave0_gate --rate 0   # in the fair world (A11)

``--rate`` defaults to the study's 0.04, the world the M1 report records.
"""

from __future__ import annotations

import argparse
import sys

from optspread.config import STUDY_R, MarketConfig
from optspread.gates import wave0_gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--rate", type=float, default=STUDY_R, help="the chain's rate r")
    args = parser.parse_args()
    report = wave0_gate(args.episodes, args.seed, market=MarketConfig(r=args.rate))
    print(report.render())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
mock_phase_shift_batch.py
========================
Generate mock photon arrival-time lists for *any* of the three source
count rate regimes (high, low, limit) defined in
``magnetar_modulation_utils.py``.

The script is fully CLI-driven so it can be scheduled on a cluster batch
system or run interactively from the command line.

Examples
--------
Generate 1 million trial events in the **high-rate** regime and save the
accepted list to *high_events.npy*:

>>> python mock_phase_shift_batch.py --rate high --n-events 1_000_000 -o high_events.npy

Generate a **limit-rate** file with a reproducible RNG seed and let the
script pick the default output name (*limit_rate_events.npy*):

>>> python mock_phase_shift_batch.py --rate limit --seed 42 --n-events 2_000_000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from magnetar_modulation_utils import simulate_events, RATES

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:  # noqa: D401 – simple wrapper
    """Return parsed command‑line arguments."""
    p = argparse.ArgumentParser(
        prog="mock_phase_shift_batch.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Generate mock photon events modulated by rotation & precession. "
            "The intrinsic pulse shape is a single‑sine with an adjustable "
            "acceptance profile."
        ),
    )

    p.add_argument(
        "--rate",
        choices=list(RATES),
        default="high",
        help="Source count‑rate regime.",
    )
    p.add_argument(
        "--n-events",
        type=int,
        default=int(1e5),
        metavar="N",
        help="Number of *trial* events before acceptance–rejection.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Output filename (.npy). If omitted a name is derived from --rate.",
    )
    return p.parse_args()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:  # noqa: D401 – entry point
    args = parse_args()

    rng = np.random.default_rng(args.seed)
    events = simulate_events(args.n_events, args.rate, rng=rng)

    outfile = args.out or Path(f"{args.rate}_rate_events.npy")
    np.save(outfile, events)

    print(
        f"[+] Saved {events.size} accepted events "
        f"({args.n_events:,} trials) -> {outfile.resolve()!s}"
    )


if __name__ == "__main__":  # pragma: no cover – standard boilerplate
    main()

#!/usr/bin/env python
"""grid_z2_map.py

Produce a colour map of :math:`Z_4^2` as a function of *trial spin period*
*P* and *trial modulation period* *T* for **fixed** modulation amplitude and
phase.  This helps locate the neighbourhood of the global maximum which
should lie close to (P_true, T_true).

The script expects a CSV file with a single column of **barycentric** photon
arrival times (seconds).  It creates one PNG image next to the input file
named ``<csvstem>_Z2map.png``.

Parameters controlling the search range and resolution can be adjusted via
command-line options.

"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from magnetar_modulation_utils import (
    A_TRUE,
    P_TRUE,
    PSI_DEG,
    T_TRUE,
    z2_binned,
)

DEG2RAD = np.pi / 180.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csvfile",
        type=Path,
        help="CSV file with one column: photon arrival times [s] (barycentric)",
    )
    parser.add_argument(
        "--p-min",
        type=float,
        default=P_TRUE - 0.01,
        help="Minimum trial spin period (seconds)",
    )
    parser.add_argument(
        "--p-max",
        type=float,
        default=P_TRUE + 0.01,
        help="Maximum trial spin period (seconds)",
    )
    parser.add_argument(
        "--p-steps",
        type=int,
        default=200,
        help="Number of grid points in P direction",
    )
    parser.add_argument(
        "--t-min",
        type=float,
        default=(T_TRUE - 12_000) / 1000.0,
        help="Minimum trial modulation period (kiloseconds)",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=(T_TRUE + 12_000) / 1000.0,
        help="Maximum trial modulation period (kiloseconds)",
    )
    parser.add_argument(
        "--t-steps",
        type=int,
        default=200,
        help="Number of grid points in T direction",
    )
    return parser.parse_args()


def worker(args: tuple[np.ndarray, float, np.ndarray]) -> np.ndarray:
    """Evaluate Z² for *all* trial P for one fixed T.

    We keep this function top‑level so it can be pickled by :pyclass:`multiprocessing.Pool`.
    """
    times, a_sin_term, p_grid = args
    # Apply phase‑shift correction for this fixed T on the fly
    corrected = times - a_sin_term
    return np.array([z2_binned(corrected, p) for p in p_grid])


def main() -> None:
    args = parse_args()

    times = np.loadtxt(args.csvfile, dtype=float)
    print(f"Loaded {len(times):,} photons from {args.csvfile.name}")

    p_grid = np.linspace(args.p_min, args.p_max, args.p_steps)
    t_grid = np.linspace(args.t_min * 1000.0, args.t_max * 1000.0, args.t_steps)

    # Precompute SIN term that depends on T only – massive speed‑up!
    sin_phase = np.sin(2.0 * np.pi * (times[:, None] / t_grid[None, :]) - PSI_DEG * DEG2RAD)

    with mp.Pool() as pool:
        z2_rows = pool.map(
            worker,
            [
                (times, A_TRUE * sin_phase[:, i], p_grid)
                for i in range(t_grid.size)
            ],
        )

    z2_map = np.vstack(z2_rows)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        z2_map,
        origin="lower",
        aspect="auto",
        extent=[p_grid[0], p_grid[-1], t_grid[0] / 1000.0, t_grid[-1] / 1000.0],
        cmap="inferno",
    )
    ax.set_xlabel("Trial spin period P [s]")
    ax.set_ylabel("Trial modulation period T [ks]")
    fig.colorbar(im, ax=ax, label=r"$Z_4^2$")

    # Mark the injected truth for reference
    ax.scatter([P_TRUE], [T_TRUE / 1000.0], c="cyan", marker="x", s=60)

    out_png = args.csvfile.with_name(args.csvfile.stem + "_Z2map.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    print(f"Saved plot → {out_png}")


if __name__ == "__main__":  # pragma: no cover
    main()

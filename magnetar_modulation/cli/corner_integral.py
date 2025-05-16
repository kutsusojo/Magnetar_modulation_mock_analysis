# -----------------------------------------------------------------------------
# corner_mock_integral_PTA.py
# -----------------------------------------------------------------------------
"""Corner plot of *integral* Z²ₙ projections.

This refactored version:
* imports **all shared constants** and `z2_binned` directly from
  ``magnetar_modulation_utils`` (no more duplicated numbers);
* wraps the top‑level logic in ``main()`` and protects it with the standard
  ``if __name__ == "__main__":`` guard so multiprocessing works on Windows;
* adds type hints and a CLI flag ``--save`` to pick the output filename.
"""
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import corner  # pip install corner

from magnetar_modulation_utils import (
    P_TRUE,
    T_TRUE,
    A_TRUE,
    PSI_DEG,
    NHARM,
    NBIN,
    z2_binned,
)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def compute_z2_cube(
    events: np.ndarray,
    p_grid: np.ndarray,
    t_grid: np.ndarray,
    a_grid: np.ndarray,
) -> np.ndarray:
    """Return Z² cube with axes (P, T, A)."""
    cube = np.empty((p_grid.size, t_grid.size, a_grid.size), dtype=float)
    for i, p in enumerate(p_grid):
        for j, t in enumerate(t_grid):
            for k, a in enumerate(a_grid):
                # For the *integral* projection we integrate Z² over phase
                z2 = z2_binned(events, p, nbin=NBIN, nharm=NHARM)
                cube[i, j, k] = z2  # placeholder; real logic may differ
    return cube

# -----------------------------------------------------------------------------
# CLI + main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Integral Z² corner plot (P, T, A projections)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "events",
        type=Path,
        help="Input .npy file with photon arrival times (output of mock_phase_shift_batch.py)",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=Path("corner_integral.png"),
        metavar="PNG",
        help="Output filename for the corner plot.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    events = np.load(args.events)

    # Example coarse grids – adapt as needed
    p_grid = np.linspace(P_TRUE * 0.999, P_TRUE * 1.001, 50)
    t_grid = np.linspace(T_TRUE * 0.8, T_TRUE * 1.2, 30)
    a_grid = np.linspace(A_TRUE * 0.5, A_TRUE * 1.5, 25)

    zcube = compute_z2_cube(events, p_grid, t_grid, a_grid)

    # Collapse the cube to 2‑D histograms for corner.py
    data = np.vstack([
        zcube.max(axis=(1, 2)),  # max over (T, A) → P axis
        zcube.max(axis=(0, 2)),  # max over (P, A) → T axis
        zcube.max(axis=(0, 1)),  # max over (P, T) → A axis
    ]).T

    fig = corner.corner(
        data,
        labels=["P [s]", "T [s]", "A [s]"],
        truths=[P_TRUE, T_TRUE, A_TRUE],
        show_titles=True,
        title_fmt="{:.3e}",
    )
    fig.savefig(args.save, dpi=300, bbox_inches="tight")
    print(f"[+] Corner plot saved to {args.save}")


if __name__ == "__main__":
    main()
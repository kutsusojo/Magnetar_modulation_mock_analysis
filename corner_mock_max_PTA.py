# -----------------------------------------------------------------------------
# corner_mock_max_PTA.py
# -----------------------------------------------------------------------------
"""Corner plot of *maximum* Z²ₙ projections (same structure as above).

Only the projection method differs; code is otherwise identical so we
factor out shared helpers.
"""
from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
import corner

from magnetar_modulation_utils import (
    P_TRUE,
    T_TRUE,
    A_TRUE,
    PSI_DEG,
    NHARM,
    NBIN,
    z2_binned,
)

# Re‑use compute_z2_cube from the first script via an import‑hack if both
# scripts live in the same directory.
# from corner_mock_integral_PTA import compute_z2_cube


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Max‑projection Z² corner plot (P, T, A axes)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("events", type=Path, help="Input .npy event file")
    p.add_argument("--save", type=Path, default=Path("corner_max.png"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    events = np.load(args.events)

    # Coarse grids (could be shared via a small helper)
    p_grid = np.linspace(P_TRUE * 0.999, P_TRUE * 1.001, 50)
    t_grid = np.linspace(T_TRUE * 0.8, T_TRUE * 1.2, 30)
    a_grid = np.linspace(A_TRUE * 0.5, A_TRUE * 1.5, 25)

    zcube = compute_z2_cube(events, p_grid, t_grid, a_grid)  # type: ignore

    data = np.vstack([
        zcube.max(axis=(1, 2)),
        zcube.max(axis=(0, 2)),
        zcube.max(axis=(0, 1)),
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
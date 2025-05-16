# -----------------------------------------------------------------------------
# make_Z2_slices.py
# -----------------------------------------------------------------------------
"""Generate PNG frame sequences of Z² slices and store them in sub-folders.

* Adds an ``if __name__ == "__main__"`` guard for Windows safety.
* Imports `z2_binned` + constants from the shared utils module.
* Writes slices for the **(P, A) over T**, **(T, A) over P**, and the new
  **(P, T) over A** planes so that *all three* corner subplots can be
  animated.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt

from magnetar_modulation_utils import (
    P_TRUE,
    T_TRUE,
    A_TRUE,
    NHARM,
    NBIN,
    z2_binned,
)

# -----------------------------------------------------------------------------

OUT_DIR = Path("frames")  # base directory for all frame sets
OUT_DIR.mkdir(exist_ok=True)


def save_frame(img: np.ndarray, fname: Path) -> None:
    plt.imsave(fname, img, cmap="viridis", origin="lower")


def worker(args):  # multiprocessing worker must be top‑level picklable
    events, p, t, a, idx, plane = args
    # Dummy image → replace with real Z² map generation
    img = np.random.random((200, 200))
    out = OUT_DIR / plane / f"frame_{idx:04d}.png"
    out.parent.mkdir(exist_ok=True)
    save_frame(img, out)
    return str(out)


def parse_args():
    p = argparse.ArgumentParser(description="Generate Z² slice frames")
    p.add_argument("events", type=Path, help="Input .npy event file")
    p.add_argument("--nproc", type=int, default=4, help="Parallel workers")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    events = np.load(args.events)

    p_grid = np.linspace(P_TRUE * 0.999, P_TRUE * 1.001, 30)
    t_grid = np.linspace(T_TRUE * 0.8, T_TRUE * 1.2, 24)
    a_grid = np.linspace(A_TRUE * 0.5, A_TRUE * 1.5, 20)

    jobs = []

    # (P, A) for each T
    for idx, t in enumerate(t_grid):
        jobs.append((events, p_grid, t, a_grid, idx, "PA"))

    # (T, A) for each P
    for idx, p in enumerate(p_grid):
        jobs.append((events, p, t_grid, a_grid, idx, "TA"))

    # (P, T) for each A  ← **new**
    for idx, a in enumerate(a_grid):
        jobs.append((events, p_grid, t_grid, a, idx, "PT"))

    with mp.Pool(args.nproc) as pool:
        list(pool.imap_unordered(worker, jobs))

    print(f"[+] Frames saved under {OUT_DIR.resolve()}")


if __name__ == "__main__":
    mp.freeze_support()  # safe‑guard on Windows
    main()
"""

magnetar_modulation_utils.py
===============

Shared utilities and physical constants for the pulse-search analysis
scripts that accompany the SN 1987A / 1E 2259+586 timing project.

All scripts in this repository import *only* from this module to
avoid copy-pasting functions such as :pyfunc:`z2_binned`.  That makes
maintenance and unit-testing significantly easier.

Author
------
sojo (Chushu Qu) on May 16, 2025

"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------
# Physical parameters injected into the mock data sets
# ---------------------------------------------------------------------

#: “True” spin period of the pulsar (seconds)
P_TRUE: float = 6.979_146_122_382_579
#: “True” modulation period (seconds)
T_TRUE: float = 17_340.0
#: Amplitude of the phase‑shift modulation (seconds)
A_TRUE: float = 0.25 * P_TRUE
#: Modulation phase angle (deg)
PSI_DEG: float = 30.0

# ---------------------------------------------------------------------
# Z² parameters
# ---------------------------------------------------------------------

#: Number of harmonics summed in the Z² periodogram.
NHARM: int = 4
#: Number of phase bins for the binned variant of Z².
NBIN: int = 32

# rates in counts s⁻¹
RATES = {"high": 20.0, "low": 2.0, "limit": 0.2}

__all__ = ["P_TRUE", "T_TRUE", "A_TRUE", "PSI_DEG", "NHARM", "NBIN", "RATES", "sample_phase_sin", "simulate_events", "z2_binned"]

# -----------------------------------------------------------------------------
# Phase‑sampling helpers
# -----------------------------------------------------------------------------

def sample_phase_sin(
    rng: np.random.Generator,
    baseline: float = 1.0,
    amp: float = 0.9,
) -> float:
    """Return a single phase \(\phi\in[0,1)\) drawn from
    :math:`f(\phi)=\text{baseline}+\text{amp}\,\sin 2\pi\phi`.

    Rejection sampling is used; the call is vectorised outside the
    function for performance.
    """
    if amp > baseline:
        raise ValueError("PDF must be non-negative everywhere (amp <= baseline)")

    max_val = baseline + amp
    while True:
        phi = rng.random()
        if rng.random() * max_val <= baseline + amp * np.sin(2.0 * np.pi * phi):
            return phi

# -----------------------------------------------------------------------------
# Event‑time simulator
# -----------------------------------------------------------------------------

def simulate_events(
    n_events: int,
    rate_label: str,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate *approximately* ``n_events`` photon arrival times for the
    specified *rate_label*.

    Parameters
    ----------
    n_events
        Number of trial events drawn **before** pulse-profile acceptance.
    rate_label
        One of ``'high'``, ``'low'``, or ``'limit'``.
    rng
        Optional `numpy.random.Generator` for reproducibility.

    Returns
    -------
    ndarray
        1-D array of accepted arrival times (seconds).
    """
    if rng is None:
        rng = np.random.default_rng()

    if rate_label not in RATES:
        raise KeyError(f"Unknown rate label '{rate_label}'. Expected one of {list(RATES)}")

    mean_rate = RATES[rate_label]

    # Draw inter‑arrival times from an exponential distribution
    dt = rng.exponential(1.0 / mean_rate, size=n_events)
    times = np.cumsum(dt, dtype=np.float64)

    # Pulse modulation via acceptance–rejection
    phase = (times % P_TRUE) / P_TRUE
    amp = 0.9 if rate_label != "limit" else 0.5  # make the faint case a bit flatter
    weight = 1.0 + amp * np.sin(2.0 * np.pi * phase)
    keep = rng.random(n_events) < weight / weight.max()

    return times[keep]


def z2_binned(
    times: Sequence[float],
    period: float,
    *,
    nbin: int = NBIN,
    nharm: int = NHARM,
) -> float:
    """Return *Z*²ₙ for an **unweighted** set of event times.

    The implementation follows the “binned” definition from Buccheri et al.
    (1983, *A&A 128, 245*).  It is ~20x faster than the unbinned variant
    for large *N* because expensive sin/cos evaluations are done only once
    per bin.

    Parameters
    ----------
    times
        Photon arrival times (seconds, barycentric).
    period
        Trial spin period *s*.
    nbin, nharm
        Number of phase bins and harmonics respectively.

    Returns
    -------
    float
        The *Z*²ₙ statistic.

    Notes
    -----
    The statistic scales roughly linearly with *N* so absolute values are
    meaningful only when *N* is fixed.  For searches use the same photon
    list throughout, or normalise appropriately.
    """
    times = np.asfarray(times)
    phase = (times / period) % 1.0
    counts, _ = np.histogram(phase, nbin, range=(0.0, 1.0))
    n = counts.sum()

    if n == 0:
        return 0.0

    omega = 2.0 * math.pi / nbin
    a = np.zeros(nharm)
    b = np.zeros(nharm)

    idx = np.arange(nbin)

    # Vectorised dot‑product → much faster than the nested loops
    for k in range(1, nharm + 1):
        ang = omega * k * idx
        cos_ang = np.cos(ang)
        sin_ang = np.sin(ang)

        a[k - 1] = (counts * cos_ang).sum()
        b[k - 1] = (counts * sin_ang).sum()

    # Normalisation follows the Buccheri definition
    z2 = ((a ** 2 + b ** 2).sum() / (2.0 * n)) * nbin ** 2
    return float(z2)

# -*- coding: utf-8 -*-
"""
source/detuning.py
====================
The anisotropy-corrected closed form for the effect of resonance
detuning on the axion-coupling Fisher information (main.tex Sec. 2.D,
Fig. 5). Derived at kappa=0 (frozen amplitude), delta0->0 (linear-
response) leading order, for the X-GHZ probe: Var(Sx)=N^2 (Heisenberg,
the probe's fiducial axis), Var(Sy)=N (standard-quantum-limit only),
Cov(Sx,Sy)=0, giving

    F_delta(Delta T)/F_delta(0)
        = sinc^2(Delta T/2) * [cos^2(Delta T/2) + sin^2(Delta T/2)/N]

which reduces to the textbook Rabi/Ramsey sinc^2(Delta T/2) power-
broadening law at N=1 (where the anisotropy vanishes). This is a
leading-order result, NOT exact at finite delta0 (see main.tex/
code/README.md for the Bloch-Siegert-type deviation this project found
and documented at the "numerically convenient" delta0=0.25).

Author: prepared for Le Bin Ho (Tohoku University) -- Manus/Manu4.
"""

import numpy as np


def full_info_ratio_analytic(DeltaT, N):
    """Anisotropy-corrected closed form F_delta(Delta T)/F_delta(0),
    valid in the delta0->0 (linear-response) limit at kappa=0. Reduces
    to sinc^2(Delta T/2) at N=1."""
    theta = np.asarray(DeltaT, dtype=float)
    sinc2 = (np.sinc(theta / (2 * np.pi))) ** 2  # = [sin(theta/2)/(theta/2)]^2
    return sinc2 * (np.cos(theta / 2) ** 2 + np.sin(theta / 2) ** 2 / N)

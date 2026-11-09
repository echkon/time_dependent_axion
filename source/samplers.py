# -*- coding: utf-8 -*-
"""
source/samplers.py
====================
Trajectory samplers for `pang_jordan.time_resolved_ensemble`/`endpoint`:
functions of the form field_sampler(rng) -> complex array of length
nsteps+1, the drive direction z(t)=x(t)+i*y(t) on a uniform grid over
[0,T]. Two families:

  - `wiener_sampler`: the phenomenological Wiener-phase-diffusion toy
    model (Sec. 2.2 of main.tex) -- z(t) = exp(i*alpha(t)), |z|=1 always,
    alpha(t) a Wiener process started from a FIXED, KNOWN fiducial phase
    alpha0=0. This is the model used for Fig. 1 (and the left panel of
    Fig. 2, and the left panel of Fig. 5).

  - `make_detuned_sampler`: wraps ANY existing sampler with a
    deterministic detuning phase ramp e^{-i*Delta*t} on top (Sec. 2.D of
    main.tex, "Departure from resonance"). Used for Fig. 5.

The physically-grounded axion field sampler (z(t)=zeta(t), variable
modulus) lives in `source/axion_field.py` instead, since it needs the
axion-mass/coupling physics from that module.

Author: prepared for Le Bin Ho (Tohoku University) -- Manus/Manu4.
"""

import numpy as np


def sample_phase_path(kappa, T, nsteps, rng, alpha0=0.0):
    """One realization of alpha(t) on a uniform grid of nsteps+1 points
    over [0, T], via Euler-Maruyama integration of d(alpha) = kappa*dW,
    starting from the FIXED fiducial phase alpha0=0 (the QFIM is
    alpha0-independent at kappa=0, so this choice is WLOG)."""
    dt = T / nsteps
    alphas = np.empty(nsteps + 1)
    alphas[0] = alpha0
    dW = rng.normal(0.0, np.sqrt(dt), size=nsteps)
    for i in range(nsteps):
        alphas[i + 1] = alphas[i] + kappa * dW[i]
    return alphas


def wiener_sampler(kappa, T, nsteps, alpha0=0.0):
    """Returns a field_sampler(rng) -> exp(i*alpha(t)) closure, ready to
    hand to `pang_jordan.time_resolved_ensemble`/`endpoint`."""

    def sampler(rng_):
        alphas = sample_phase_path(kappa, T, nsteps, rng_, alpha0=alpha0)
        return np.exp(1j * alphas)

    return sampler


def make_detuned_sampler(base_sampler, Delta, T, nsteps):
    """Wrap a field_sampler(rng)->complex array (length nsteps+1, on the
    grid t=linspace(0,T,nsteps)) so its output also carries a
    DETERMINISTIC detuning phase ramp e^{-i*Delta*t} (main.tex Sec. 2.D:
    valid RWA extension for |Delta| << omega_q, m_a)."""
    t_grid = np.linspace(0.0, T, nsteps + 1)
    phase_ramp = np.exp(-1j * Delta * t_grid)

    def sampler(rng_):
        return base_sampler(rng_) * phase_ramp

    return sampler

# -*- coding: utf-8 -*-
"""
source/axion_field.py
=======================
The physically-grounded, time-dependent stochastic axion field (Foster,
Rodd & Safdi, Phys. Rev. D 97, 123006 (2018), arXiv:1711.10489, Eqs.
(8)-(10) "sum of plane waves" construction; coherence-time/Rayleigh-
envelope statistics cross-checked against Centers et al., Nat. Commun.
12, 7321 (2021), arXiv:1905.13650, Eq. (2)), plus the axion-photon
coupling g_agg <-> encoding-angle conversion (Chen et al., PRD 110,
115021 (2024), Eqs. (2.6)-(3.15)) and the KSVZ/DFSZ benchmark curves --
all ported unchanged from Manu3's own
Codes/code_no_noise/{axion_reach_results/axion_coupling.py,
topo/run/mass_scan_bayes.py}, kept in this one module so every
run_fig*.py script that needs the physical field imports from a single
place.

zeta(t) = (1/sqrt(2*n_modes)) * sum_j alpha_j exp(i[m_a*v_j^2/2*t+phi_j])
is the normalized complex field envelope, in the frame rotating at the
qubit's resonance (tuned to m_a): <|zeta(t)|^2> -> 1 as n_modes->infinity,
|zeta(t)| asymptotically Rayleigh(1)-distributed. Feed this directly as
a `field_sampler` to `source.pang_jordan.time_resolved_ensemble`/
`endpoint` (see `field_sampler_for_mass` below for the ready-made
closure); UNLIKE the Wiener toy model's z(t)=exp(i*alpha(t)), |zeta(t)|
itself fluctuates -- physically, the DM field's amplitude, not just its
phase, varies on the coherence timescale.

Author: prepared for Le Bin Ho (Tohoku University) -- Manus/Manu4.
"""

import numpy as np

ALPHA_FS = 1 / 137.036
HBAR_EVS = 6.582119569e-16  # eV*s
OMEGA_PER_UEV = 1e-6 / HBAR_EVS * 1e-6  # rad/us per ueV of mass


# ---------------------------------------------------------------
# KSVZ / DFSZ benchmark coupling curves
# ---------------------------------------------------------------
def g_KSVZ(m_a_ueV):
    return (ALPHA_FS / (2 * np.pi)) * 1.92 * m_a_ueV / 5.69e12


def g_DFSZ(m_a_ueV):
    return (ALPHA_FS / (2 * np.pi)) * abs(8 / 3 - 1.92) * m_a_ueV / 5.69e12


# ---------------------------------------------------------------
# DM coherence time and coupling -> encoding-angle conversion
# ---------------------------------------------------------------
def tau_DM_us(m_a_ueV, v0=1e-3):
    """DM coherence time in microseconds, in ONE fixed convention.

    For the SHM mode sum of sample_axion_envelope() the normalized
    autocorrelation of zeta is exactly

        C(t) = (1 - i m_a v0^2 t / hbar)^(-3/2),

    whose spectral width is dw = m_a v0^2 / hbar.  We use the standard
    full-cycle convention (Foster-Rodd-Safdi 2018; Centers et al. 2021),

        tau_DM = 2*pi/dw = 2*pi*hbar / (m_a v0^2),

    i.e. tau_DM = 4135.7 us * (1 ueV / m_a) * (1e-3 / v0)^2.
    Conventions differ by O(2pi); the integral convention
    int_0^inf |C|^2 dt = hbar/(m_a v0^2) is 1/(2*pi) of this value, and
    |C| reaches 1/e at 0.27 tau_DM.  When the residual coherence over a
    given interval is what matters, use C(t) itself, not tau_DM.

    NOTE: the interrogation time is NOT capped at tau_DM anywhere in
    this work; it is set by the qubit coherence time T2.  The field
    trajectory zeta(t) is sampled over the whole interval [0, T], so
    decorrelation within a shot is included exactly.
    """
    return 4135.7 * (1.0 / m_a_ueV) * (1e-3 / v0) ** 2


def delta_axion(g_agg, m_a_ueV, tau_us=None, T2_qubit_us=100.0,
                 B0_T=5.0, kappa=1.0, C_pF=0.1, d_um=100.0, rho_DM=0.45):
    """Per-qubit encoding rotation angle delta = eta*tau for a trial
    axion-photon coupling g_agg [GeV^-1] at mass m_a_ueV [ueV]. Exactly
    linear in g_agg (p_ge1 ~ g_agg^2, delta = sqrt(p_ge1) ~ |g_agg|)."""
    if tau_us is None:
        tau_us = T2_qubit_us  # interrogation time is set by T2, not capped at tau_DM
    p_ge1 = (0.11
             * (g_agg / 1e-10) ** 2
             * (m_a_ueV / 1.0) ** -1
             * (B0_T / 1.0) ** 2
             * (tau_us / 100.0) ** 2
             * kappa ** 2
             * (C_pF / 0.1)
             * (d_um / 100.0) ** 2
             * (rho_DM / 0.45))
    return np.sqrt(p_ge1)


def g_agg_from_delta(delta, m_a_ueV, tau_us=None, T2_qubit_us=100.0,
                      B0_T=5.0, kappa=1.0, C_pF=0.1, d_um=100.0, rho_DM=0.45):
    """Inverse of delta_axion."""
    if tau_us is None:
        tau_us = T2_qubit_us  # interrogation time is set by T2, not capped at tau_DM
    p_ge1 = delta ** 2
    factor = (0.11
              * (m_a_ueV / 1.0) ** -1
              * (B0_T / 1.0) ** 2
              * (tau_us / 100.0) ** 2
              * kappa ** 2
              * (C_pF / 0.1)
              * (d_um / 100.0) ** 2
              * (rho_DM / 0.45))
    return 1e-10 * np.sqrt(p_ge1 / factor)


def s_and_k_for_mass(g_true, m_a_ueV, T_us):
    """s_true = delta_axion(g_true,...)/T (EXACTLY linear in g_true), and
    k_true = s_true/g_true (chain rule; independent of g_true itself,
    only of m_a_ueV and T_us). F_g = k_true^2 * F_s follows from Fisher
    information transforming quadratically under the linear
    reparametrization s = k*g -- this is how every g_agg-parametrized
    figure (2 middle/right panel, 4) converts a Fisher information
    computed w.r.t. the propagator-amplitude rate s into Fisher
    information w.r.t. the physical coupling g_agg itself."""
    s_true = delta_axion(g_true, m_a_ueV, tau_us=T_us) / T_us
    k_true = s_true / g_true
    return s_true, k_true


# ---------------------------------------------------------------
# Stochastic axion field envelope: sum-of-modes (Foster-Rodd-Safdi)
# ---------------------------------------------------------------
def sample_axion_envelope(m_a_ueV, T_us, nsteps, n_modes, rng, v0=1e-3):
    """One realization of the normalized complex field envelope zeta(t)
    on a uniform grid of nsteps+1 points over [0, T_us] (microseconds).
    Earth's ~232 km/s boost through the halo is neglected (isotropic
    Galactic-frame SHM only, v0=1e-3 <-> ~300 km/s dispersion)."""
    t = np.linspace(0.0, T_us, nsteps + 1)
    omega_a = OMEGA_PER_UEV * m_a_ueV
    vx = rng.normal(0.0, v0, n_modes)
    vy = rng.normal(0.0, v0, n_modes)
    vz = rng.normal(0.0, v0, n_modes)
    v2 = vx ** 2 + vy ** 2 + vz ** 2
    omega_j = omega_a * v2 / 2.0
    alpha_j = rng.rayleigh(scale=1.0, size=n_modes)
    phi_j = rng.uniform(0.0, 2 * np.pi, size=n_modes)
    phase = np.outer(t, omega_j) + phi_j[None, :]
    zeta = (alpha_j[None, :] * np.exp(1j * phase)).sum(axis=1) / np.sqrt(2.0 * n_modes)
    return zeta


def field_sampler_for_mass(m_a_ueV, T_us, nsteps, n_modes):
    """Returns a field_sampler(rng) -> zeta(t) closure at fixed
    (m_a_ueV, T_us, nsteps, n_modes), ready to hand to
    `source.pang_jordan.time_resolved_ensemble`/`endpoint`."""

    def sampler(rng_):
        return sample_axion_envelope(m_a_ueV, T_us, nsteps, n_modes, rng_)

    return sampler

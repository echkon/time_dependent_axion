# -*- coding: utf-8 -*-
"""
run/run_weakexp_check.py
========================
Verification of Appendix "Weak-coupling expansion of the ensemble QFI".

Predicts the coefficient of theta^2 in R_g = |Q^tr/Q^ens - 1| from two
statistics of the field alone,

    R_g  ~=  (K - 1) * rho * theta^2 * N/(N+1),

    rho = E[(int_0^T Re zeta dt)^2] / T^2        (coherent fraction)
    K   = E[k^4]/E[k^2]^2,  k = int_0^T Re zeta dt   (kurtosis)

and compares it with R_g measured from the exact 2^N ensemble QFI.
The three non-SHM rows deliberately move rho and K to test the law
over a factor ~50 in the predicted coefficient.

Reproduces Table I of the manuscript.  Runtime ~10 min for the full set.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from source.axion_field import sample_axion_envelope, OMEGA_PER_UEV
from source.pang_jordan import endpoint

T, NS = 100.0, 100
TGRID = np.linspace(0.0, T, NS + 1)
NR_STAT = 20000            # realizations for rho and K
SEED_STAT, SEED_QFI = 5, 424242


def shm(m_a=1.0, n_modes=200):
    return lambda rng: sample_axion_envelope(m_a, T, NS, n_modes, rng)


def single_mode(amp):
    """One SHM mode with a prescribed amplitude law; E[|zeta|^2] = 1."""
    def smp(rng):
        A = amp(rng)
        v = rng.normal(0.0, 1e-3, 3)
        w = OMEGA_PER_UEV * 1.0 * (v @ v) / 2.0
        return A * np.exp(1j * (w * TGRID + rng.uniform(0, 2 * np.pi))) / np.sqrt(2.0)
    return smp


def field_stats(sampler, nr=NR_STAT, seed=SEED_STAT):
    rng = np.random.default_rng(seed)
    k = np.array([np.trapezoid(sampler(rng).real, TGRID) for _ in range(nr)])
    m2, m4 = (k ** 2).mean(), (k ** 4).mean()
    return m2 / T ** 2, m4 / m2 ** 2          # rho, kurtosis


def measure(sampler, N, theta, n_traj, seed=SEED_QFI):
    r = endpoint(sampler, theta / (N * T), T, NS, N, n_traj,
                 np.random.default_rng([seed, N, int(theta * 1e4)]))
    return abs(r["full"] / r["ensemble"] - 1.0) / theta ** 2


CASES = [("SHM, m_a=1 ueV",   shm(1.0),  3, 0.06, 4000),
         ("SHM, m_a=1 ueV",   shm(1.0),  5, 0.06, 4000),
         ("SHM, m_a=1 ueV",   shm(1.0),  7, 0.06, 4000),
         ("SHM, m_a=1 ueV",   shm(1.0),  9, 0.06,  700),
         ("SHM, m_a=10 ueV",  shm(10.0), 5, 0.06, 4000),
         ("heavy-tailed",     single_mode(lambda r: r.exponential(1.0)), 5, 0.01, 6000),
         ("fixed amplitude",  single_mode(lambda r: np.sqrt(2.0)),       5, 0.06, 4000)]

if __name__ == "__main__":
    print(f"{'field':<18}{'N':>3}{'rho':>8}{'K':>8}{'pred':>9}{'meas':>9}{'ratio':>8}")
    for name, smp, N, theta, ntr in CASES:
        rho, K = field_stats(smp)
        pred = (K - 1.0) * rho * N / (N + 1.0)
        meas = measure(smp, N, theta, ntr)
        print(f"{name:<18}{N:>3}{rho:>8.3f}{K:>8.2f}{pred:>9.2f}{meas:>9.2f}{meas/pred:>8.2f}")

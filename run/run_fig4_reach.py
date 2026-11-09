# -*- coding: utf-8 -*-
"""
run/run_fig4_reach.py
=======================
COMPUTE step for Fig. 4 (main.tex): axion-photon coupling reach vs. mass
-- projected 1-sigma sensitivity sigma_g(m_a) on g_agg for N qubits, at
Ref. Ho2026Manu3's scan exposure, plus a simulated Bayesian-style
targeted recovery at m_a=1 ueV. Saves raw results to
../data/fig4_reach.json; plot with `plot_fig4_reach.py`.

>>> TO INCREASE N: edit N_LIST below and re-run. <<<
Cost note: dimension is 2**N; this script runs len(N_LIST) x
len(masses_ueV) full endpoint() calculations. N up to ~9 is comfortable
on a laptop; N=10-11 will take noticeably longer per mass point.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np

from source.pang_jordan import endpoint
from source.axion_field import (g_KSVZ, g_DFSZ, tau_DM_us, field_sampler_for_mass,
                                 s_and_k_for_mass)
from source.paths import data_path

# ---------------------------------------------------------------
# Tunable parameters -- edit and re-run
# ---------------------------------------------------------------
N_LIST = [3,5,10]                 # qubit numbers to sweep (was [1, 5])
N_TRY_SCAN = 1.05e6                # scan exposure per mass bin
N_TRY_TARGET = 3.156e11            # targeted single-mass exposure
N_RUNS = 5                         # synthetic recovery draws
n_modes = 200
nsteps = 100
n_traj = 300
masses_ueV = np.geomspace(0.1, 10.0, 7)
m_target = 1.0                     # mass for the targeted-recovery demo
N_target = 5                       # qubit number for the targeted-recovery demo
SEED = 20260829

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    results = {"masses_ueV": masses_ueV.tolist(), "N_TRY_SCAN": N_TRY_SCAN,
               "N_TRY_TARGET": N_TRY_TARGET, "N_list": N_LIST, "seed": SEED,
               "g_KSVZ": [g_KSVZ(m) for m in masses_ueV],
               "g_DFSZ": [g_DFSZ(m) for m in masses_ueV],
               "points": {}}

    print("=== g_agg reach sweep (physically-grounded time-dependent field) ===")
    for N in N_LIST:
        results["points"][str(N)] = {"full": [], "ensemble": []}
        for m_a in masses_ueV:
            T_us = 100.0
            g_ref = g_KSVZ(m_a)
            s_true, k_true = s_and_k_for_mass(g_ref, m_a, T_us)
            sampler = field_sampler_for_mass(m_a, T_us, nsteps, n_modes)
            r = endpoint(sampler, s_true, T_us, nsteps, N, n_traj, rng)
            F_g_f = k_true ** 2 * r["full"]
            F_g_e = k_true ** 2 * r["ensemble"]

            sigma_g_f = 1.0 / np.sqrt(max(F_g_f, 1e-300) * N_TRY_SCAN)
            sigma_g_e = 1.0 / np.sqrt(max(F_g_e, 1e-300) * N_TRY_SCAN)

            results["points"][str(N)]["full"].append({
                "m_a_ueV": m_a, "T_us": T_us, "F_g": F_g_f, "sigma_g": sigma_g_f, "s_true": s_true})
            results["points"][str(N)]["ensemble"].append({
                "m_a_ueV": m_a, "T_us": T_us, "F_g": F_g_e, "sigma_g": sigma_g_e})
            print(f"N={N}  m_a={m_a:6.3f} ueV  sigma_g^ensemble={sigma_g_e:.3e} "
                  f"({sigma_g_e/g_ref:.1f}x KSVZ)  full/ensemble-1={100*(F_g_f/F_g_e-1):+.4f}%")

    # --- targeted single-mass recovery demonstration ---
    T_target = 100.0
    g_true_target = g_KSVZ(m_target)
    s_true_t, k_true_t = s_and_k_for_mass(g_true_target, m_target, T_target)
    sampler_t = field_sampler_for_mass(m_target, T_target, nsteps, n_modes)
    r_t = endpoint(sampler_t, s_true_t, T_target, nsteps, N_target, 500, rng)
    F_g_e_t = k_true_t ** 2 * r_t["ensemble"]
    sigma_target = 1.0 / np.sqrt(F_g_e_t * N_TRY_TARGET)
    draws = g_true_target + rng.normal(0.0, sigma_target, N_RUNS)
    g_hat_mean, g_hat_std = float(np.mean(draws)), float(np.std(draws))
    print(f"\n=== targeted recovery: m_a={m_target} ueV, N={N_target}, N_try={N_TRY_TARGET:.3g} ===")
    print(f"g_true = {g_true_target:.4e}, g_hat = {g_hat_mean:.4e} +- {sigma_target:.4e} "
          f"(bias = {100*(g_hat_mean/g_true_target-1):.2f}%)")
    results["targeted"] = {"m_a_ueV": m_target, "T_us": T_target, "N": N_target,
                            "N_try": N_TRY_TARGET, "g_true": g_true_target,
                            "F_g": F_g_e_t, "sigma_g": sigma_target,
                            "g_hat_mean": g_hat_mean, "g_hat_std": g_hat_std}

    out = data_path("fig4_reach.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")

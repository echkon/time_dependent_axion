# -*- coding: utf-8 -*-
"""
run/run_fig2_time_resolved.py
===============================
COMPUTE step for Fig. 2 (main.tex): full-information / ensemble /
fixed-basis-CFI Fisher information resolved at 30 checkpoints WITHIN a
single shot (0 < t <= T), for two cases -- the Wiener toy model
(kappa*T=2) and the physically-grounded axion field (m_a=1 ueV, KSVZ) --
plus the resulting projected coupling sensitivity sigma_g(t) for the
physical-field case. Saves raw results to ../data/fig2_time_resolved.json;
plot with `plot_fig2_time_resolved.py`.

>>> TO INCREASE N: edit N below and re-run. <<<
Cost note: dimension is 2**N; runtime here also scales with nsteps and
n_traj (both cases loop through nsteps time-steps per trajectory, times
n_traj trajectories). N up to ~9-10 is comfortable; N=11-12 will be
noticeably slower per checkpoint. Try lowering n_traj or nsteps first if
you just want a quick look at a larger N before committing to a full run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np

from source.pang_jordan import time_resolved_ensemble
from source.samplers import wiener_sampler
from source.axion_field import field_sampler_for_mass, tau_DM_us, delta_axion, g_KSVZ, s_and_k_for_mass
from source.paths import data_path

# ---------------------------------------------------------------
# Tunable parameters -- edit and re-run
# ---------------------------------------------------------------
N = 10                              # qubit number (was 5 in the paper)
nsteps = 200
n_traj = 400                       # Monte Carlo trajectories per case
n_checkpoints = 30                 # time-resolution within the shot

# Case A: Wiener toy model
kappa_A, delta_A, T_A = 2.0, 0.25, 1.0

# Case B: physically-grounded field
m_a_B = 1.0                        # ueV
n_modes_B = 200
N_TRY_TARGET = 3.156e11            # targeted single-mass search exposure

SEED = 9182736

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    checkpoint_steps = np.unique(np.linspace(1, nsteps, n_checkpoints).astype(int))
    t_frac = checkpoint_steps / nsteps

    results = {"checkpoint_t_over_T": t_frac.tolist(), "N": N, "nsteps": nsteps,
               "n_traj": n_traj, "seed": SEED}

    print(f"=== Case A: Wiener toy model (kappa*T={kappa_A*T_A:.1f}, N={N}) ===")
    sampler_A = wiener_sampler(kappa_A, T_A, nsteps)
    res_A = time_resolved_ensemble(sampler_A, delta_A, T_A, nsteps, N, n_traj, rng, checkpoint_steps)
    results["wiener"] = {k: v.tolist() for k, v in res_A.items()}
    for i in [0, len(checkpoint_steps) // 2, -1]:
        print(f"t/T={t_frac[i]:.2f}  full={res_A['full'][i]:.3f}  "
              f"ensemble={res_A['ensemble'][i]:.3f}  cfi_fixed={res_A['cfi_fixed'][i]:.3f}")

    print(f"\n=== Case B: physically-grounded field (m_a={m_a_B} ueV, N={N}) ===")
    T_B = 100.0
    g_B = g_KSVZ(m_a_B)
    s_B, k_B = s_and_k_for_mass(g_B, m_a_B, T_B)
    sampler_B = field_sampler_for_mass(m_a_B, T_B, nsteps, n_modes_B)

    res_B_s = time_resolved_ensemble(sampler_B, s_B, T_B, nsteps, N, n_traj, rng, checkpoint_steps)
    # Convert Fisher information w.r.t. the rate parameter s into Fisher
    # information w.r.t. the physical coupling g_agg: F_g = k_B^2 * F_s.
    res_B = dict(res_B_s)
    for key in ("full", "full_err", "ensemble", "cfi_fixed", "ensemble_raw", "cfi_fixed_raw"):
        res_B[key] = (k_B ** 2) * res_B_s[key]

    def safe_sigma(F, N_try):
        F = np.where(np.isnan(F), np.nan, np.maximum(F, 1e-300))
        with np.errstate(divide="ignore", invalid="ignore"):
            return 1.0 / np.sqrt(F * N_try)

    sigma_g_full = safe_sigma(res_B["full"], N_TRY_TARGET)
    sigma_g_ensemble = safe_sigma(res_B["ensemble"], N_TRY_TARGET)
    sigma_g_cfi = safe_sigma(res_B["cfi_fixed"], N_TRY_TARGET)
    res_B["sigma_g_full"] = sigma_g_full
    res_B["sigma_g_ensemble"] = sigma_g_ensemble
    res_B["sigma_g_cfi_fixed"] = sigma_g_cfi

    results["field"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res_B.items()}
    results["field"]["T_us"] = T_B
    results["field"]["s_true"] = s_B
    results["field"]["k_true"] = k_B
    results["field"]["g_true"] = g_B
    results["field"]["N_TRY_TARGET"] = N_TRY_TARGET
    for i in [0, len(checkpoint_steps) // 2, -1]:
        print(f"t/T={t_frac[i]:.2f}  F_g_full={res_B['full'][i]:.3e}  "
              f"F_g_ensemble={res_B['ensemble'][i]:.3e}  "
              f"sigma_g_ensemble={sigma_g_ensemble[i]:.3e} ({sigma_g_ensemble[i]/g_B:.1f}x KSVZ)")

    out = data_path(f"fig2_time_resolved_{N}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")

# -*- coding: utf-8 -*-
"""
run/run_paper_fig5_dephasing.py
=================================
COMPUTE step for paper Fig. 5 -- how independent per-qubit dephasing
during the interrogation limits the collective enhancement.

Produces two things:
  (a) sigma_g(N) at fixed mass for several dephasing strengths gamma*T,
      over a wide range of N, from the closed form of source/dephasing.py;
  (b) the projected sensitivity vs. mass at a large fixed N, with and
      without dephasing -- the practical caveat on the noiseless Fig. 3.
Plus, with --validate, exact dense 2^N points at small N to confirm the
closed form on the same footing as the figure itself.

>>> TO CHANGE the dephasing grid or N range: edit GAMMA_T_LIST / N_GRID. <<<
Cost is independent of N (the dense validation is not, hence its own flag).

Output: ../data/paper_fig5_dephasing.json
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np

from source.axion_field import (g_KSVZ, g_DFSZ, tau_DM_us,
                                 field_sampler_for_mass, s_and_k_for_mass)
from source.dephasing import (generator_coeffs, qfi_dephased, crossover_N,
                              qfi_dephased_dense, optimal_N)
from source.paths import data_path

# ---------------------------------------------------------------
GAMMA_T_LIST = [0.0, 0.001, 0.005, 0.02]   # dimensionless dephasing gamma*T
# extended to 1000 so that the x axis reaches a full decade at 10^3 and
# the LOCATED minimum of every gamma*T curve falls inside the plotted
# range (the gamma*T = 0.001 one sits near 500)
N_GRID = sorted(set(list(range(1, 21)) +
                    [int(round(x)) for x in np.geomspace(21, 1000, 36)]))
N_FIXED = 100                 # network size for the reach-vs-mass panel
m_fixed = 1.0                 # ueV, for the sigma_g(N) panel
N_TRY_SCAN = 1.05e6
n_modes, nsteps, n_traj = 200, 100, 300
masses_ueV = np.geomspace(0.1, 10.0, 7)
T2_us = 100.0
SEED = 20260916
# dense validation (only with --validate; cost is 2^N)
VAL_N = [3, 5, 7]
VAL_GAMMA_T = [0.02]
VAL_NTRAJ = 3
VAL_NSTEPS = 60


def coeffs_for_mass(m_a, n_realizations, nst, seed_extra=0):
    """(c_x, c_y, c_z) for each field realization at one mass."""
    T_us = T2_us
    s_true, k_true = s_and_k_for_mass(g_KSVZ(m_a), m_a, T_us)
    sampler = field_sampler_for_mass(m_a, T_us, nst, n_modes)
    cs = []
    for j in range(n_realizations):
        rng = np.random.default_rng([SEED, seed_extra, j])
        cs.append(generator_coeffs(sampler, s_true, T_us, nst, rng))
    return np.array(cs), T_us, s_true, k_true


if __name__ == "__main__":
    t0 = time.time()
    res = {"gamma_T_list": GAMMA_T_LIST, "N_grid": N_GRID, "N_fixed": N_FIXED,
           "m_fixed_ueV": m_fixed, "masses_ueV": masses_ueV.tolist(),
           "N_TRY_SCAN": N_TRY_SCAN, "n_modes": n_modes, "nsteps": nsteps,
           "n_traj": n_traj, "seed": SEED,
           "g_KSVZ": [g_KSVZ(m) for m in masses_ueV],
           "g_DFSZ": [g_DFSZ(m) for m in masses_ueV],
           "crossover_N": {str(g): crossover_N(g) for g in GAMMA_T_LIST}}

    # ---- (a) sigma_g vs N at fixed mass -------------------------------
    print(f"=== (a) sigma_g vs N at m_a = {m_fixed} ueV ===")
    cs, T_us, s_true, k_true = coeffs_for_mass(m_fixed, n_traj, nsteps)
    print(f"    <c_x^2> = {np.mean(cs[:,0]**2):.4e}   "
          f"<c_y^2+c_z^2> = {np.mean(cs[:,1]**2 + cs[:,2]**2):.4e}")
    res["panel_a"] = {}
    for gT in GAMMA_T_LIST:
        sig = []
        for N in N_GRID:
            F_g = k_true ** 2 * np.mean([qfi_dephased(c, N, gT) for c in cs])
            sig.append(1.0 / np.sqrt(F_g * N_TRY_SCAN))
        res["panel_a"][str(gT)] = sig
        # LOCATE the minimum of the full expression instead of assuming N_c:
        # N_c solves only the collective-term condition, and the linear term
        # shifts the true stationary point upward.
        cx2 = float(np.mean(cs[:, 0] ** 2))
        cyz2 = float(np.mean(cs[:, 1] ** 2 + cs[:, 2] ** 2))
        res.setdefault("optimal_N", {})[str(gT)] = optimal_N(cx2, cyz2, gT)
        print(f"    gamma*T={gT:<7} N_c={crossover_N(gT):8.1f}  "
              f"N_opt={res['optimal_N'][str(gT)]}   "
              f"sigma_g(N=1)/sigma_g(N=last) = {sig[0]/sig[-1]:6.2f}")

    # ---- (b) reach vs mass at fixed large N ---------------------------
    print(f"\n=== (b) sensitivity vs mass at N = {N_FIXED} ===")
    res["panel_b"] = {}
    for gT in GAMMA_T_LIST:
        sig = []
        for i_m, m_a in enumerate(masses_ueV):
            cs_m, T_m, s_m, k_m = coeffs_for_mass(m_a, n_traj, nsteps, seed_extra=i_m + 1)
            F_g = k_m ** 2 * np.mean([qfi_dephased(c, N_FIXED, gT) for c in cs_m])
            sig.append(1.0 / np.sqrt(F_g * N_TRY_SCAN))
        res["panel_b"][str(gT)] = sig
        print(f"    gamma*T={gT:<7} sigma_g/g_KSVZ: "
              f"{sig[0]/res['g_KSVZ'][0]:6.2f}x -> {sig[-1]/res['g_KSVZ'][-1]:5.2f}x")

    # ---- optional exact dense validation ------------------------------
    if "--validate" in sys.argv:
        print("\n=== dense 2^N validation of the closed form ===")
        val = []
        T_v = T2_us
        s_v, k_v = s_and_k_for_mass(g_KSVZ(m_fixed), m_fixed, T_v)
        sampler = field_sampler_for_mass(m_fixed, T_v, VAL_NSTEPS, n_modes)
        for gT in VAL_GAMMA_T:
            for N in VAL_N:
                ex, cf = [], []
                for j in range(VAL_NTRAJ):
                    rng = np.random.default_rng([SEED, 999, j])
                    zt = sampler(rng)
                    rng2 = np.random.default_rng([SEED, 999, j])
                    c = generator_coeffs(sampler, s_v, T_v, VAL_NSTEPS, rng2)
                    ex.append(qfi_dephased_dense(zt, s_v, T_v, VAL_NSTEPS, N, gT))
                    cf.append(qfi_dephased(c, N, gT))
                # convert QFI w.r.t. s into QFI w.r.t. g, as elsewhere
                e, f = float(np.mean(ex)) * k_v ** 2, float(np.mean(cf)) * k_v ** 2
                val.append({"N": N, "gamma_T": gT, "F_g_exact": e,
                            "F_g_closed_form": f, "rel_err": f / e - 1})
                print(f"    N={N:<3} gamma*T={gT:<6} exact={e:.6e} "
                      f"closed form={f:.6e}  rel.err={f/e-1:+.2e}")
        res["validation"] = val

    out = data_path("paper_fig5_dephasing.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nTotal {time.time()-t0:.1f}s.  Saved {out}")

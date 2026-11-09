# -*- coding: utf-8 -*-
"""
run/run_fig3_crossover.py
===========================
COMPUTE step for Fig. 3 (main.tex): the linear- vs. nonlinear-response
crossover -- full-information/ensemble Fisher-information advantage vs.
the injected rotation-angle scale s (fixed m_a=1 ueV), for N qubits.
This is the mechanism DIAGNOSTIC explaining why the two estimators
coincide at the physical KSVZ signal strength (see Fig. 2's physical
field panel for the direct demonstration). Saves raw results to
../data/fig3_crossover.json; plot with `plot_fig3_crossover.py`.

>>> TO INCREASE N: edit N_LIST below and re-run. <<<
Cost note: dimension is 2**N; this script runs a full endpoint()
calculation (nsteps time-steps x n_traj trajectories) at EVERY point of
s_grid x N_LIST, so total cost scales as len(N_LIST) x len(s_grid) x
(cost per N). N up to ~9 is comfortable; consider trimming s_grid or
n_traj first if pushing N higher.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np

from source.pang_jordan import endpoint
from source.axion_field import field_sampler_for_mass, delta_axion, g_KSVZ
from source.paths import data_path

# ---------------------------------------------------------------
# Tunable parameters -- edit and re-run
# ---------------------------------------------------------------
N_LIST = [1, 5, 9, 11]          # qubit numbers to sweep -- these are the N that
                                   # produced the committed ../data/fig3_crossover.json
                                   # (run on the cluster). Fig. 2(a) of main.tex reads
                                   # the N values out of that file, so keep the two in
                                   # step: changing this list means re-running to the
                                   # end, or the plot will still show the old N.
                                   # An earlier attempt at [3, 5, 10] was interrupted
                                   # before json.dump and left the data untouched
                                   # (see the traceback in nohup.out).
m_fix, T_fix = 1.0, 100.0          # ueV, us
n_modes = 200
nsteps = 100
n_traj = 300
s_grid = np.array([1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5,
                    1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 0.6])
SEED = 20260829

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    s_physical = delta_axion(g_KSVZ(m_fix), m_fix, tau_us=T_fix) / T_fix

    results = {"s_grid": s_grid.tolist(), "N_list": N_LIST,
               "m_fix": m_fix, "T_fix": T_fix, "s_physical_m1ueV": s_physical,
               "seed": SEED, "full": {}, "ensemble": {}}

    print("=== linear- vs. nonlinear-response crossover ===")
    for N in N_LIST:
        full_vals, ensemble_vals = [], []
        for s in s_grid:
            sampler = field_sampler_for_mass(m_fix, T_fix, nsteps, n_modes)
            r = endpoint(sampler, s, T_fix, nsteps, N, n_traj, rng)
            full_vals.append(r["full"])
            ensemble_vals.append(r["ensemble"])
        results["full"][str(N)] = full_vals
        results["ensemble"][str(N)] = ensemble_vals
        print(f"N={N}: " + "  ".join(
            f"s={s:.0e}:{100*(f/e-1):+.0f}%" if e else f"s={s:.0e}:n/a"
            for s, f, e in zip(s_grid, full_vals, ensemble_vals)))

    out = data_path("fig3_crossover.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")

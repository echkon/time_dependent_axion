# -*- coding: utf-8 -*-
"""
run/run_fig5_detuning_Nscan.py
================================
EXTRA COMPUTE step for paper Fig. 5(b): the PHYSICAL sum-of-modes field
plus detuning, now for SEVERAL network sizes instead of the single
N = 7 of `run_fig5_detuning.py` part 4.

Why this exists.  Panel (b) used to show three estimators at one N.
They agree to ~1e-8 in normalized form, because theta = N eta(g) T
~ 5e-5 is deep in the weak-signal regime (Sec. III B), so the three
curves carry one statement between them. Showing ONE estimator at
several N instead makes panel (b) directly comparable with panel (a),
which is the monochromatic, deterministic closed form at the same N.

Only the trajectory-resolved QFI is computed, via the collective
closed form of Appendix B (`endpoint_tr_collective`), whose cost does
not grow with N -- the same route Fig. 4 uses.

Two sampling modes are produced:
  "indep" : the rng stream runs on across the DeltaT grid, exactly as
            part 4 of run_fig5_detuning.py does. Each DeltaT point sees
            its own field realizations, so the curve carries sampling
            scatter.
  "crn"   : common random numbers -- the rng is reseeded to the SAME
            value at every DeltaT and every N, so all points use one
            fixed set of realizations. Differences along the curve are
            then pure detuning response, and differences between curves
            are pure N dependence.

Output: ../data/fig5_detuning_Nscan.json
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np

from source.collective import endpoint_tr_collective
from source.samplers import make_detuned_sampler
from source.axion_field import field_sampler_for_mass, tau_DM_us, delta_axion, g_KSVZ
from source.paths import data_path

N_LIST = [3, 5, 10]                 # matches panel (a)
nsteps = 200
n_traj = 300
m_a = 1.0                           # ueV
n_modes = 200
SEED = 73310501
DeltaT_grid = np.array([0.0, 0.2, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5,
                        3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0])

if __name__ == "__main__":
    T = 100.0
    g = g_KSVZ(m_a)
    s = delta_axion(g, m_a, tau_us=T) / T
    base = field_sampler_for_mass(m_a, T, nsteps, n_modes)
    print(f"m_a = {m_a} ueV, T = {T} us, s = {s:.4e} /us, "
          f"eta*T = {s*T:.4e}, n_traj = {n_traj}")

    out = {"DeltaT_grid": DeltaT_grid.tolist(), "N_list": N_LIST,
           "T": T, "s": s, "m_a_ueV": m_a, "n_traj": n_traj, "seed": SEED,
           "indep": {}, "crn": {}}

    for mode in ("indep", "crn"):
        t0 = time.time()
        rng = np.random.default_rng(SEED)
        for N in N_LIST:
            vals = []
            for DT in DeltaT_grid:
                if mode == "crn":
                    rng = np.random.default_rng(SEED)
                sampler = make_detuned_sampler(base, DT / T, T, nsteps)
                r = endpoint_tr_collective(sampler, s, T, nsteps, N, n_traj, rng)
                vals.append(float(r["full"]))
            out[mode][str(N)] = vals
            v = np.array(vals) / vals[0]
            print(f"  [{mode}] N={N:<3} "
                  + "  ".join(f"{x:.3f}" for x in v))
        print(f"  [{mode}] done in {time.time()-t0:.1f} s")

    with open(data_path("fig5_detuning_Nscan.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("saved", data_path("fig5_detuning_Nscan.json"))

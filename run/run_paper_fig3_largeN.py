# -*- coding: utf-8 -*-
"""
run/run_paper_fig3_largeN.py
==============================
COMPUTE step for paper Fig. 3 (projected g_agg sensitivity vs. mass) at
LARGE qubit number.

>>> TO ADD A NEW N: append it to N_LIST and re-run. Only the N values <<<
>>> that have no data file yet are computed; the rest are skipped.     <<<

ONE FILE PER N. Each N writes its own
    ../data/paper_fig3_largeN_N<N>.json
so the work is never repeated. Running with N_LIST = [1, 5, 10, 20] and
then later with N_LIST = [1, 5, 10, 20, 40] computes only N = 40. Pass
--force to recompute N values that already have a file (needed if you
change n_traj, nsteps, n_modes or the mass grid, since those are what the
stored numbers depend on -- the file records them so you can check).

Why this is not just run_fig4_reach.py with a bigger N_LIST: that script
calls `pang_jordan.endpoint`, which builds the trajectory-averaged density
matrix as a dense 2^N x 2^N array. At N = 14 one such matrix is already
~4 GB and two are needed, so the dense path stops there no matter how long
you are willing to wait. This script instead evaluates the closed form of
Appendix B (implemented in `source/collective.py`), in which the whole
N-dependence of the trajectory-resolved QFI is analytic,

    F = 4 [ c_x^2 N^2 + (c_y^2 + c_z^2) N ],

and nothing larger than a 2x2 matrix is ever allocated. Runtime is
therefore INDEPENDENT of N: N = 50 costs exactly what N = 5 costs, and the
total is set only by n_traj * nsteps * len(masses) * (number of NEW N).

The closed form is exact, not an approximation -- it agrees with the dense
construction to ~1e-16 relative for N = 1..9. What it does NOT give is the
trajectory-AVERAGED QFI or the fixed-measurement CFI, which genuinely need
the 2^N state. That is harmless here: at the physical KSVZ signal strength
the two QFIs agree to ~1e-4 % (Sec. III A, Fig. 2a), which is why Fig. 3
plots a single curve per N.

Monte Carlo note: field trajectories are seeded PER MASS, not per
(mass, N), so every N sees the identical set of field realizations. The
curves differ only through the exact N-dependence and not through
independent Monte Carlo scatter -- which is also what makes the
one-file-per-N scheme safe, since a previously computed N would come out
bit-identical if recomputed.

Output: ../data/paper_fig3_largeN_N<N>.json   (one per N)
        Plot them all with plot_paper_fig3_sensitivity_vs_mass.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np

from source.axion_field import (g_KSVZ, g_DFSZ, tau_DM_us,
                                 field_sampler_for_mass, s_and_k_for_mass)
from source.collective import endpoint_tr_collective
from source.paths import data_path

# ---------------------------------------------------------------
# Tunable parameters -- edit and re-run
# ---------------------------------------------------------------
N_LIST = [1, 5, 10, 20, 30, 40, 50, 80, 100]   # >>> cost does NOT grow with N <<<
N_TRY_SCAN = 1.05e6                # scan exposure per mass bin
n_modes = 200                      # sum-of-modes field realizations
nsteps = 100                       # time steps per trajectory
n_traj = 300                       # Monte Carlo field realizations per point
masses_ueV = np.geomspace(0.1, 10.0, 7)
T2_us = 100.0                      # qubit coherence time
SEED = 20260829


def compute_one_N(N):
    """All mass points for a single N. Returns the dict written to disk."""
    rows = []
    for i_m, m_a in enumerate(masses_ueV):
        T_us = T2_us
        g_ref = g_KSVZ(m_a)
        s_true, k_true = s_and_k_for_mass(g_ref, m_a, T_us)
        sampler = field_sampler_for_mass(m_a, T_us, nsteps, n_modes)
        # seeded per mass only -> identical trajectories for every N
        rng = np.random.default_rng([SEED, i_m])
        r = endpoint_tr_collective(sampler, s_true, T_us, nsteps, N, n_traj, rng)
        F_g = k_true ** 2 * r["full"]
        F_g_err = k_true ** 2 * r["full_err"]
        sigma_g = 1.0 / np.sqrt(max(F_g, 1e-300) * N_TRY_SCAN)
        rows.append({"m_a_ueV": float(m_a), "T_us": T_us, "F_g": F_g,
                     "F_g_err": F_g_err, "sigma_g": sigma_g, "s_true": s_true})
        print(f"    m_a={m_a:6.3f} ueV   sigma_g={sigma_g:.3e} GeV^-1"
              f"   ({sigma_g/g_ref:7.1f}x KSVZ)")
    return {"N": N,
            "masses_ueV": masses_ueV.tolist(),
            "N_TRY_SCAN": N_TRY_SCAN,
            "seed": SEED, "n_modes": n_modes, "nsteps": nsteps, "n_traj": n_traj,
            "T2_us": T2_us,
            "method": "collective closed form, Appendix B (source/collective.py)",
            "g_KSVZ": [g_KSVZ(m) for m in masses_ueV],
            "g_DFSZ": [g_DFSZ(m) for m in masses_ueV],
            "points": rows}


if __name__ == "__main__":
    force = "--force" in sys.argv
    print("=== projected g_agg sensitivity vs. mass, large-N (collective) ===")
    print(f"N_LIST = {N_LIST};  {n_traj} trajectories x {nsteps} steps per point")
    print(f"{'recomputing every N (--force)' if force else 'skipping N values that already have a data file'}\n")

    t_start = time.time()
    done, skipped = [], []
    for N in N_LIST:
        out = data_path(f"paper_fig3_largeN_N{N}.json")
        if os.path.exists(out) and not force:
            skipped.append(N)
            print(f"  N={N:<4} already present -> skipped ({os.path.basename(out)})")
            continue
        print(f"  N={N:<4} computing...")
        t0 = time.time()
        res = compute_one_N(N)
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        done.append(N)
        print(f"    -> {os.path.basename(out)}  ({time.time()-t0:.1f}s)\n")

    print(f"\nComputed {done if done else 'nothing new'}"
          f"{'; skipped ' + str(skipped) if skipped else ''}"
          f".  Total {time.time()-t_start:.1f}s.")
    print("Now run:  python3 plot_paper_fig3_sensitivity_vs_mass.py")

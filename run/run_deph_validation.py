# -*- coding: utf-8 -*-
"""
run/run_deph_validation.py
============================
Direct validation of the closed-form noisy QFI, Eq. (deph) of the main
text, against the FULL noisy evolution.

The reference integrates
    d rho/dt = -(i/hbar)[H_DM(t;g), rho] + gamma sum_k (sx_k rho sx_k - rho)
jointly with d/dg of the same equation, then evaluates the SLD QFI. The
Kraus step rho -> (1-p) rho + p sx rho sx with p = [1-exp(-2 gamma dt)]/2
is the EXACT solution of the dissipator over one step, so the only error
is the Hamiltonian/dissipator splitting, which is O(dt^2) per step and is
checked by varying nsteps.

Usage:  python3 run_deph_validation.py <chunk>
        chunk = "small"  -> N = 3, 5   (gamma*T sweep at g = g_KSVZ)
        chunk = "n7"     -> N = 7      (same sweep)
        chunk = "gscan"  -> N = 5, coupling sweep at gamma*T = 0.02
Results are merged into ../data/deph_validation.json
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from source.dephasing import qfi_dephased_dense, generator_coeffs, qfi_dephased
from source.axion_field import field_sampler_for_mass, tau_DM_us, delta_axion, g_KSVZ
from source.paths import data_path

M_A, NSTEPS, NTRAJ, SEED = 1.0, 50, 8, 4242
GT_LIST = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
G_RATIOS = [1.0, 1e2, 1e3, 1e4, 1e5]
OUT = data_path("deph_validation.json")


def load():
    return json.load(open(OUT)) if os.path.exists(OUT) else {}


def save(d):
    json.dump(d, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    chunk = sys.argv[1] if len(sys.argv) > 1 else "small"
    T = 100.0
    s0 = delta_axion(g_KSVZ(M_A), M_A, tau_us=T) / T
    samp = field_sampler_for_mass(M_A, T, NSTEPS, 200)
    zts = [samp(np.random.default_rng([SEED, j])) for j in range(NTRAJ)]
    cs = [generator_coeffs(samp, s0, T, NSTEPS, np.random.default_rng([SEED, j]))
          for j in range(NTRAJ)]
    res = load()
    res.setdefault("meta", {"m_a_ueV": M_A, "T_us": T, "nsteps": NSTEPS,
                            "n_traj": NTRAJ, "seed": SEED,
                            "gamma_T_list": GT_LIST, "g_ratios": G_RATIOS})

    if chunk in ("small", "n7"):
        Ns = [3, 5] if chunk == "small" else [7]
        res.setdefault("gt_sweep", {})
        for N in Ns:
            t0 = time.time()
            row = {}
            for gT in GT_LIST:
                ex = np.mean([qfi_dephased_dense(z, s0, T, NSTEPS, N, gT) for z in zts])
                cf = np.mean([qfi_dephased(c, N, gT) for c in cs])
                row[str(gT)] = {"exact": float(ex), "closed": float(cf),
                                "rel_err": float(cf / ex - 1.0)}
                print(f"  N={N} gT={gT:<6} rel.err={cf/ex-1:+.3e}", flush=True)
            res["gt_sweep"][str(N)] = row
            print(f"  -> N={N} done in {time.time()-t0:.0f}s", flush=True)

    elif chunk == "gscan":
        N = 5
        res.setdefault("g_sweep", {})
        for gr in G_RATIOS:
            s = s0 * gr
            cs_g = [generator_coeffs(samp, s, T, NSTEPS, np.random.default_rng([SEED, j]))
                    for j in range(NTRAJ)]
            ex = np.mean([qfi_dephased_dense(z, s, T, NSTEPS, N, 0.02) for z in zts])
            cf = np.mean([qfi_dephased(c, N, 0.02) for c in cs_g])
            res["g_sweep"][f"{gr:.0e}"] = {"exact": float(ex), "closed": float(cf),
                                           "rel_err": float(cf / ex - 1.0),
                                           "delta0": float(s * T)}
            print(f"  g/g_KSVZ={gr:<8.0e} delta0={s*T:.2e}  rel.err={cf/ex-1:+.3e}", flush=True)
    save(res)
    print("saved", OUT)

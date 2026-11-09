# -*- coding: utf-8 -*-
"""
run/plot_paper_fig6_deph_validation.py
========================================
APPENDIX FIGURE: accuracy of the closed-form noisy QFI, Eq. (deph),
against the full noisy evolution.

The reference integrates the Lindblad equation
    d rho/dt = -(i/hbar)[H_DM(t;g), rho] + gamma sum_k (sx_k rho sx_k - rho)
jointly with its g-derivative and evaluates the SLD QFI; the Kraus step
used for the dissipator is its exact single-step solution, so the only
error is the Hamiltonian/dissipator splitting.

Plotted: |closed form / exact - 1| vs gamma*T for N = 3, 5, 7, with the
band of gamma*T values used in the main text shaded.

Input : ../data/deph_validation.json
Output: ../figs/paper_fig6_deph_validation.eps
        Manuscript slot: manu/figures/deph_validation.eps
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from source.paths import data_path, figs_path
from source import paper_style as ps

GT_USED = (0.001, 0.02)     # range of gamma*T shown in the dephasing figure


def blend(color, alpha, bg="white"):
    c = np.array(mcolors.to_rgb(color)); b = np.array(mcolors.to_rgb(bg))
    return tuple(alpha * c + (1.0 - alpha) * b)


if __name__ == "__main__":
    ps.use_paper_style()
    d = json.load(open(data_path("deph_validation.json")))
    sweep = d["gt_sweep"]
    N_list = sorted(int(k) for k in sweep)
    colors = ps.n_ramp(len(N_list))

    fig, (ax,) = ps.fig_wide(
        1, width=ps.COL_W, height=ps.COL_W / 1.75,
        margins=dict(left=0.66, right=0.08, bottom=0.42, top=0.08, gap=0.0))

    ax.axvspan(*GT_USED, color=blend("#0072B2", 0.16), lw=0, zorder=0)
    ax.text(np.sqrt(GT_USED[0] * GT_USED[1]), 0.93, "used here",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=6.5, color="0.25")

    for i, N in enumerate(N_list):
        row = sweep[str(N)]
        gts = np.array(sorted(float(k) for k in row))
        err = np.array([abs(row[f"{g:g}"]["rel_err"]) for g in gts])
        ax.plot(gts, err, marker=ps.n_marker(i), ls=ps.n_linestyle(i),
                color=colors[i], mfc="none", mew=1.0, ms=4.2, zorder=3,
                label=rf"$N={N}$")
        print(f"  N={N}: " + "  ".join(f"{g:g}:{e:.1e}" for g, e in zip(gts, err)))

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\gamma T$")
    ax.set_ylabel(r"$|Q_g^{\rm deph}/Q_g^{\rm exact}-1|$")
    ax.legend(loc="upper left", fontsize=6.5, frameon=False,
              handlelength=3.0, handletextpad=0.4, labelspacing=0.2,
              borderaxespad=0.4)
    ps.save(fig, figs_path("paper_fig6_deph_validation"), tight=False)
    plt.close(fig)

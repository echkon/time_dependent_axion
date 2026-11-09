# -*- coding: utf-8 -*-
"""
run/plot_paper_fig4_detuning.py
=================================
PAPER FIG. 5 (main.tex): effect of a residual detuning on the
trajectory-resolved QFI.

ONE panel, single column: the PHYSICAL sum-of-modes field only.

    Q_g^tr(T; dw) / Q_g^tr(T; 0)   vs   dw*T       for N = 3, 5, 10

The old panel (a) -- a kappa=0, deterministic, monochromatic toy at
delta0 = 1e-4, used to validate a closed-form sinc response -- was
dropped together with that closed form. It tested a configuration the
paper never studies, and the one prediction that distinguished it from
the textbook Ramsey law (an N-dependent narrowing) does not survive
contact with the physical field: see the N=3 vs N=10 curves here, which
agree to ~1%.

Common random numbers: every detuning and every N uses the SAME 300
field realizations, so the shape along a curve is the detuning response
rather than sampling scatter, and the spread between curves is the N
dependence alone.

Input : ../data/fig5_detuning_Nscan.json
Output: ../figs/paper_fig4_detuning.eps
        Manuscript slot: manu/figures/off_resonance.eps
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from source.paths import data_path, figs_path
from source import paper_style as ps

if __name__ == "__main__":
    ps.use_paper_style()

    with open(data_path("fig5_detuning_Nscan.json")) as f:
        nsc = json.load(f)

    DT = np.array(nsc["DeltaT_grid"], dtype=float)
    # spanning N = 3 ... 100 makes the N-independence a much stronger
    # statement than 50...100 alone would; N = 5 and 80 are computed and
    # lie on the same curve, and are omitted only to keep the panel legible
    N_SHOW = [3, 10, 50, 100]
    N_list = [N for N in nsc["N_list"] if N in N_SHOW]
    colors = ps.n_ramp(len(N_list))
    print(f"m_a = {nsc['m_a_ueV']} ueV, T = {nsc['T']} us, "
          f"n_traj = {nsc['n_traj']}, N = {N_list}")

    # same canvas proportions as Fig. 3
    fig, (ax,) = ps.fig_wide(
        1, width=ps.COL_W, height=ps.COL_W / 1.75,
        margins=dict(left=0.70, right=0.08, bottom=0.42, top=0.08, gap=0.0))

    for i, N in enumerate(N_list):
        y = np.array(nsc["crn"][str(N)], dtype=float)
        y = y / y[0]
        ax.plot(DT, y, marker=ps.n_marker(i), ls="--", color=colors[i],
                mfc="none", mew=1.0, ms=4.2, zorder=3, label=rf"$N={N}$")
        print(f"  N={N:<3} " + "  ".join(f"{v:.3f}" for v in y))

    ax.set_xlabel(r"$\Delta\omega\,T$")
    ax.set_ylabel(r"$Q_g^{\rm tr}(T;\Delta\omega)\,/\,Q_g^{\rm tr}(T;0)$")
    ax.set_xlim(-0.3, 8.3)
    ax.set_ylim(-0.05, 1.12)
    ax.legend(loc="upper right", fontsize=7, frameon=False,
              handlelength=2.6, handletextpad=0.5, labelspacing=0.25,
              borderaxespad=0.5)

    ps.save(fig, figs_path("paper_fig4_detuning"), tight=False)
    plt.close(fig)

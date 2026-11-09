# -*- coding: utf-8 -*-
"""
run/plot_paper_fig2_trajectory_information.py
===============================================
PAPER FIG. 3 -- "Validity of the trajectory-resolved QFI" (Sec. III B).

ONE panel, single column (never figure*):

    R_g(T) = |Q_g^tr/Q_g^ens - 1|   vs   theta = N eta(g) T

The earlier two-panel version also showed the same data against
g/g_KSVZ. That panel was dropped: theta is the variable that actually
controls the ratio, so plotting against g alone separates the curves for
a reason that is not physical (only the product N*eta*T matters) and
invites the reader to read an N dependence into the coupling axis.

The physical operating point is marked instead by a BAND, because on the
theta axis it is not a single point: theta_KSVZ = 6.5e-6 * N, so the
configurations used anywhere in this paper occupy

    theta_KSVZ in [6.5e-6, 6.5e-4]     for  N = 1 ... 100

and the whole band sits three decades below the theta ~ 0.7 threshold at
which trajectory knowledge starts to matter. That single band therefore
carries what the dropped panel carried (where the real experiment sits)
AND justifies the extrapolation to the N = 50...100 networks of Fig. 4,
where the 2^N ensemble calculation is out of reach.

WHY THE X AXIS NEEDS NO NEW RUN.
`fig3_crossover.json` was swept over the drive-rate scale s, a leftover
from the older formulation. eta(g) is EXACTLY linear in g, so s = k*g
with k independent of g, and s is numerically the same object as eta(g).
Hence theta = N*s*T identically, and `s_physical_m1ueV` stored in the
same file is s at the KSVZ coupling for m_a = 1 ueV, T = 100 us.

The y axis is the bare ratio, NOT per cent, and absolute -- at small
theta the two QFIs agree to ~1e-12 and the sign of the difference is
pure numerical noise. The flat scatter at the left IS that noise floor;
the physics is the clean power-law rise above it.

EPS has no alpha channel, so the band is a PRE-BLENDED solid colour.

Input : ../data/fig3_crossover.json
Output: ../figs/paper_fig2_trajectory_information.eps
        Manuscript slot: manu/figures/full_info_ensemble_crossover.eps
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from source.paths import data_path, figs_path
from source import paper_style as ps

N_BAND_MAX = 100          # largest network used anywhere in the paper (Fig. 4)
N_SHOW = [5, 9, 11]       # N=1 dropped: it adds a curve, not a statement


def blend(color, alpha, bg="white"):
    """Pre-blend `color` against `bg`: EPS has no alpha channel, so a
    semi-transparent patch would be written out fully opaque."""
    c = np.array(mcolors.to_rgb(color))
    b = np.array(mcolors.to_rgb(bg))
    return tuple(alpha * c + (1.0 - alpha) * b)


if __name__ == "__main__":
    ps.use_paper_style()

    with open(data_path("fig3_crossover.json")) as f:
        cross = json.load(f)

    s_grid = np.array(cross["s_grid"], dtype=float)
    s_phys = float(cross["s_physical_m1ueV"])
    T_fix = float(cross["T_fix"])
    etaT = s_phys * T_fix                       # eta(g)T at the KSVZ coupling
    N_list = [N for N in sorted(int(k) for k in cross["full"].keys())
              if N in N_SHOW]
    colors = ps.n_ramp(len(N_list))
    print(f"m_a = {cross['m_fix']} ueV, T = {T_fix} us, N = {N_list}")
    print(f"eta(g)T at KSVZ = {etaT:.4g}  ->  KSVZ band in theta: "
          f"[{etaT:.3g}, {etaT * N_BAND_MAX:.3g}]  for N = 1 ... {N_BAND_MAX}")

    # canvas pinned to the same 1.356 ratio as Fig. 4
    fig, (ax,) = ps.fig_wide(
        1, width=ps.COL_W, height=ps.COL_W / 1.75,
        margins=dict(left=0.62, right=0.08, bottom=0.42, top=0.08, gap=0.0))

    YLIM = (1e-10, 3e3)

    # ---- the physical operating band, behind everything ---------------
    ax.axvspan(etaT, etaT * N_BAND_MAX, color=blend("#0072B2", 0.16),
               lw=0, zorder=0)
    ax.text(np.sqrt(etaT * etaT * N_BAND_MAX), 0.93, "KSVZ",
            transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=6.5, color="0.25")

    # ---- the curves ---------------------------------------------------
    for i, N in enumerate(N_list):
        q_tr = np.array(cross["full"][str(N)], dtype=float)
        q_ens = np.array(cross["ensemble"][str(N)], dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            adv = np.abs(q_tr / q_ens - 1.0)
        theta = N * s_grid * T_fix
        # a marker on EVERY point: with markevery the line kinks at
        # points that carry no marker, which reads as a plotting glitch
        ax.plot(theta, adv, marker=ps.n_marker(i), ls=ps.n_linestyle(i),
                color=colors[i], mfc="none", mew=1.0, ms=4.2,
                zorder=3, label=rf"$N={N}$")
        m = adv > 1e-4
        print(f"  N={N:<3} R_g=1 at theta = "
              f"{10 ** np.interp(0.0, np.log10(adv[m]), np.log10(theta[m])):.3f}")

    # ---- the weak-signal boundary -------------------------------------
    ax.axvline(0.7, color="0.45", ls=":", lw=1.1, zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-7, 4e3)
    ax.set_ylim(*YLIM)
    ax.set_xticks([1e-6, 1e-3, 1e0, 1e3])
    ax.set_xlabel(r"$\theta=N\eta(g)T$")
    ax.set_ylabel(r"$R_g(T)$")
    ax.legend(loc="lower right", ncol=1, fontsize=6.5, frameon=False,
              handlelength=3.0, handletextpad=0.4, labelspacing=0.2,
              borderaxespad=0.4)

    ps.save(fig, figs_path("paper_fig2_trajectory_information"), tight=False)
    plt.close(fig)

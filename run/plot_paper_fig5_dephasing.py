# -*- coding: utf-8 -*-
"""
run/plot_paper_fig5_dephasing.py
==================================
PAPER FIG. 5 -- "Effect of dephasing on the collective enhancement".

  (a) sigma_g at fixed mass vs. network size N, for several dephasing
      strengths gamma*T. The noiseless curve follows the Heisenberg 1/N
      slope; each dephasing curve peels away from it near
      N_c = 1/(2 gamma T) and reverts to the standard 1/sqrt(N) slope,
      because dephasing along the fiducial axis kills the N^2 term
      (exp(-4 N gamma T)) while the linear term survives. Open markers are
      the exact dense 2^N simulation, plotted on top of the closed form.
  (b) The practical consequence for Fig. 3: projected sensitivity vs. mass
      at N = 100 for the same dephasing strengths, against the KSVZ and
      DFSZ lines.

Input : ../data/paper_fig5_dephasing.json (run_paper_fig5_dephasing.py)
Output: ../figs/paper_fig5_dephasing.png/.eps
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
    # SINGLE-COLUMN figure: main.tex loads it with the ordinary `figure`
    # environment and \linewidth. Do not make it a figure*.
    #
    # Room at 3.40 in comes from two things, not from shrinking the type:
    #   * the legend lives ABOVE the panels, so the data area is empty;
    #   * the margins are cut to the minimum the tick labels need.
    # Type stays at the project default (8 pt labels, 7 pt ticks), i.e.
    # exactly the size it is in every other figure.
    SC = 1.0
    ps.use_paper_style(scale=SC)
    with open(data_path("paper_fig5_dephasing.json")) as f:
        res = json.load(f)

    N = np.array(res["N_grid"], dtype=float)
    gTs = res["gamma_T_list"]
    colors = ps.n_ramp(len(gTs))
    m_fix = res["m_fixed_ueV"]
    g_ksvz = np.array(res["g_KSVZ"], dtype=float)
    g_dfsz = np.array(res["g_DFSZ"], dtype=float)
    masses = np.array(res["masses_ueV"], dtype=float)
    # g_KSVZ at the fixed mass used in panel (a)
    g_at_fix = float(np.interp(m_fix, masses, g_ksvz))

    # PAGE-wide figure: two panels side by side, each axes box 3 wide : 4
    # tall. main.tex loads this one with \begin{figure*} + width=\textwidth.
    # At column width the same layout is legible but cramped; here each panel
    # is ~3.3 in across, so the labels, the legend and the validation inset
    # all have room.
    fig, axes = ps.fig_wide(
        2, panel_aspect=4.0 / 3.0, width=ps.COL_W,
        margins=dict(left=0.50, right=0.03, bottom=0.40, top=0.10, gap=0.64))

    # ---- (a) sigma_g vs N -------------------------------------------
    ax = axes[0]
    for i, gT in enumerate(gTs):
        y = np.array(res["panel_a"][str(gT)], dtype=float) / g_at_fix
        lab = rf"$\gamma T={gT:g}$"
        ax.plot(N, y, ls=ps.n_linestyle(i), color=colors[i], lw=1.5, label=lab)
        # circle at the LOCATED minimum of the curve, not at N_c: N_c solves
        # only the collective-term condition and the linear term shifts the
        # true stationary point upward.
        Nopt = res.get("optimal_N", {}).get(str(gT))
        if Nopt is not None and N[0] <= Nopt <= N[-1]:
            ax.plot([Nopt], [np.interp(Nopt, N, y)], "o", color=colors[i],
                    ms=6, mfc="white", mew=1.3, zorder=5)
    # slope guides
    ref = np.array(res["panel_a"][str(gTs[0])], dtype=float)[0] / g_at_fix
    ax.plot(N, ref / N, ":", color="0.55", lw=1.0)
    ax.text(N[2], ref / N[2] * 0.62, r"$1/N$", fontsize=6.5, color="0.4",
            ha="right", va="top")
    ax.plot(N, ref / np.sqrt(N), "--", color="0.55", lw=1.0)
    ax.text(N[4], ref / np.sqrt(N[4]) * 1.30, r"$1/\sqrt{N}$", fontsize=6.5,
            color="0.4", ha="left", va="bottom")

    ax.set_xscale("log"); ax.set_yscale("log")
    # keep every decade of the sweep labelled, including 10^0 and 10^3
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\sigma_g\,/\,g_{\rm KSVZ}$")
    _y0, _y1 = ax.get_ylim()
    ax.set_ylim(_y0, _y1 * 1.35)   # headroom for the (a) tag only
    ps.panel_label(ax, "(a)")

    # ---- (b) sensitivity vs mass at fixed large N --------------------
    ax2 = axes[1]
    ax2.plot(masses, g_ksvz, "-", color="0.15", lw=1.3, label="KSVZ")
    ax2.plot(masses, g_dfsz, "--", color="0.15", lw=1.3, label="DFSZ")
    for i, gT in enumerate(gTs):
        y = np.array(res["panel_b"][str(gT)], dtype=float)
        lab = rf"$\gamma T={gT:g}$"
        ax2.plot(masses, y, ls=ps.n_linestyle(i), color=colors[i],
                 lw=1.5, label=lab)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel(r"$m_a$  [$\mu$eV]")
    ax2.set_ylabel(r"$\sigma_g$  [GeV$^{-1}$]")
    ax2.set_ylim(bottom=ax2.get_ylim()[0] / 1.8)
    # the two benchmark lines get a small legend of their own, in the empty
    # wedge below DFSZ at large mass; in-place labels there would sit on the
    # data curves
    h2, l2 = ax2.get_legend_handles_labels()
    keep = [(h, l) for h, l in zip(h2, l2) if l in ("KSVZ", "DFSZ")]
    ax2.legend([h for h, _ in keep], [l for _, l in keep], loc="lower right",
               fontsize=6.5, frameon=False, handlelength=1.6,
               handletextpad=0.5, labelspacing=0.2, borderaxespad=0.4)
    ps.panel_label(ax2, "(b)")

    # the four gamma*T entries sit INSIDE panel (a), in the empty wedge
    # below the curves at small N; panel (b) shows the same four colours
    # and line styles and keeps its own KSVZ/DFSZ legend.
    ax.legend(loc="lower left", fontsize=6, frameon=False,
              handlelength=2.8, handletextpad=0.4, labelspacing=0.22,
              borderaxespad=0.3)


    # margins are set explicitly by ps.fig_wide -- no tight_layout here
    ps.save(fig, figs_path("paper_fig5_dephasing"), tight=False)

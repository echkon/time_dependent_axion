# -*- coding: utf-8 -*-
"""
run/plot_paper_fig1_time_resolved.py
======================================
PAPER FIG. 1 -- "Time-resolved axion information" (Sec. III A).

PHYSICAL FIELD ONLY. The Wiener comparison that used to share this figure
now lives in paper Fig. 2, so that Fig. 1 answers exactly one question:
*how does the information on g_agg accumulate during the interrogation?*

  (a) F_g^tr(t), F_g^ens(t), F_g^C(t)   -- m_a = 1 ueV, KSVZ, N qubits
  (b) sigma_g(t)/g_KSVZ                 -- same run, targeted exposure

Optional inset in (a): the relative gap (F_g^tr/F_g^ens - 1), because on a
log axis the three curves lie on top of each other -- which is the physical
point, but is hard to *prove* visually. The inset shows the gap is at the
1e-4 % level rather than merely "too small to see". Set SHOW_INSET = False
to drop it.

Input : ../data/fig2_time_resolved_<N>.json  (written by run_fig2_time_resolved.py)
Output: ../figs/paper_fig1_time_resolved_N<N>.png/.eps  -- one per N found.
        The version you want in the manuscript goes to manu/figures/qfi_vs_time.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from source.paths import data_path, figs_path
from source import paper_style as ps

SHOW_INSET = True

if __name__ == "__main__":
    ps.use_paper_style()

    files = sorted(glob.glob(data_path("fig2_time_resolved_*.json")))
    if not files:
        sys.exit("No ../data/fig2_time_resolved_*.json found -- run run_fig2_time_resolved.py first.")

    for path in files:
        n_tag = int(os.path.basename(path).replace(".json", "").split("_")[-1])
        with open(path) as f:
            res = json.load(f)

        t = np.array(res["checkpoint_t_over_T"])
        N = res["N"]
        fld = res["field"]
        g_true = fld["g_true"]
        n_try = fld["N_TRY_TARGET"]

        fig, axes = ps.fig_col(2, sharex=True)

        # main.tex notation: Q = quantum Fisher information (trajectory-
        # resolved / trajectory-averaged), F = classical Fisher information
        # of the fixed measurement. Kept local to this figure: Fig. 5 still
        # uses the older all-F notation.
        EST = [("full", ps.C_TR, ps.M_TR, ps.LS_TR, r"$Q_g^{\rm tr}$"),
               ("ensemble", ps.C_ENS, ps.M_ENS, ps.LS_ENS, r"$Q_g^{\rm ens}$"),
               ("cfi_fixed", ps.C_CFI, ps.M_CFI, ps.LS_CFI, r"$F_g$")]

        # ---- (a) Fisher information vs time -------------------------
        ax = axes[0]
        for key, c, mk, ls, lab in EST:
            y = np.array(fld[key], dtype=float)
            ax.plot(t, y, marker=mk, ls=ls, color=c, label=lab,
                    mfc="none", mew=1.2, markevery=2)
        ax.set_yscale("log")
        # headroom so the upper-left legend never overlaps the rising curve
        ymax = np.nanmax([np.nanmax(np.array(fld[k], dtype=float))
                          for k, *_ in EST])
        ymin = np.nanmin(np.array(fld["full"], dtype=float))
        ax.set_ylim(ymin / 3.0, ymax * 60.0)
        ax.set_ylabel(r"Quantum inf.  [GeV$^{2}$]")
        # same legend as panel (b); the 60x headroom above keeps it clear
        ax.legend(loc="upper right", ncol=3, columnspacing=0.8,
                  borderaxespad=1.1, bbox_to_anchor=(0.68, 1.0))
        ps.panel_label(ax, "(a)")

        if SHOW_INSET:
            tr = np.array(fld["full"], dtype=float)
            ens = np.array(fld["ensemble"], dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                gap = tr / ens - 1.0          # bare ratio, NOT per cent
            ok = np.isfinite(gap)
            axin = ax.inset_axes([0.52, 0.19, 0.44, 0.34])
            axin.plot(t[ok], np.abs(gap[ok]), "-", color=ps.C_TR, lw=1.1)
            axin.set_yscale("log")
            # pin the bottom to a decade so the axis has a stated floor
            # (the data run down to ~5e-9)
            axin.set_ylim(1e-10, 1.0)
            axin.set_yticks([1e-9, 1e-6, 1e-3])
            axin.set_xlabel(r"$t/T$", fontsize=5.5, labelpad=1)
            axin.set_ylabel(r"$R_g(t)$", fontsize=6, labelpad=1)
            axin.tick_params(labelsize=5, length=1.5, pad=1)

        # ---- (b) projected sensitivity vs time ----------------------
        ax2 = axes[1]
        for key, c, mk, ls, lab in [
                ("sigma_g_full", ps.C_TR, ps.M_TR, ps.LS_TR, r"$Q_g^{\rm tr}$"),
                ("sigma_g_ensemble", ps.C_ENS, ps.M_ENS, ps.LS_ENS, r"$Q_g^{\rm ens}$"),
                ("sigma_g_cfi_fixed", ps.C_CFI, ps.M_CFI, ps.LS_CFI, r"$F_g$")]:
            y = np.array(fld[key], dtype=float) / g_true
            ax2.plot(t, y, marker=mk, ls=ls, color=c, label=lab, mfc="none",
                     mew=1.2, markevery=2)
        ax2.axhline(1.0, color="black", ls="--", lw=0.8, zorder=4)
        ax2.text(0.24, 1.06, r"$\sigma_g=g_{\rm KSVZ}$", fontsize=6, color="black",
                 transform=ax2.get_yaxis_transform(), va="bottom")
        ax2.set_yscale("log")
        ax2.set_xlabel(r"$t/T$")
        ax2.set_ylabel(r"$\sigma_g(t)\,/\,g_{\rm KSVZ}$")
        ax2.legend(loc="center right", ncol=3, columnspacing=0.8,
                   borderaxespad=1.1)
        ps.panel_label(ax2, "(b)", loc="lower left")

        fig.tight_layout()
        ps.save(fig, figs_path(f"paper_fig1_time_resolved_N{n_tag}"))
        plt.close(fig)

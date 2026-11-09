# -*- coding: utf-8 -*-
"""
run/plot_paper_fig3_sensitivity_vs_mass.py
============================================
PAPER FIG. 3 -- "Projected axion-photon coupling sensitivity versus mass"
(Sec. III C).

  sigma_g(m_a) = 1/sqrt(F_g N_try) at the scan exposure, one curve per N,
  against the KSVZ and DFSZ benchmark lines.

The figure is retitled away from the word "reach", which reads as
exclusion/discovery reach; what is plotted is a projected sensitivity.

The reconstruction star that used to sit at m_a = 1 ueV has been removed.
It was not a reconstruction in any meaningful sense: run_fig4_reach.py
produced it by drawing five Gaussian numbers around the injected coupling
with width sigma_g, so it could only ever land on the injected value and
carried no information the sensitivity curves did not already carry.

Data source, in order of preference:
  1. ../data/paper_fig3_largeN_N<N>.json, one per N, written by
     run_paper_fig3_largeN.py -- the collective closed form of Appendix B,
     so N is not capped. Every such file found is plotted, which means a
     new N appears in the figure as soon as its file exists, with no need
     to recompute the others.
  2. ../data/fig4_reach.json (run_fig4_reach.py) -- the older dense path,
     capped near N = 13, used only if no per-N files are present.
Existing constraints are overlaid the same way as in Ref. Ho2026Manu3's
Fig. 4: the ADMX and RBF/UF haloscope exclusion envelopes as filled bands
(data in ../data/bounds/, see the README there), and the CAST helioscope
and HB/AGB globular-cluster cooling bounds as horizontal lines. Anything
ABOVE a band or line is already excluded.

Output: ../figs/paper_fig3_sensitivity_vs_mass.eps
        Manuscript slot: manu/figures/g_agg_reconstruction.eps
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import json
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from source.paths import data_path, figs_path
from source import paper_style as ps

# Which N to DRAW. None = every per-N data file found. Set e.g.
# [1, 5, 10, 20, 50] to thin the figure without deleting any data --
# above N ~ 20 the curves bunch together, because sigma_g ~ 1/N.
N_SHOW = [1, 50, 80, 100]

# Existing constraints, as in Ref. Ho2026Manu3's Fig. 4. Files hold the
# AxionLimits polygon envelopes (mass [eV], g [GeV^-1]); see
# ../data/bounds/README.md.  (file, fill, edge, label, label x [ueV], label y)
EXPERIMENTS = [
    ("ADMX2018.txt", "goldenrod", "darkgoldenrod", "ADMX", 3.75, 8e-12),
    ("ADMX2021.txt", "goldenrod", "darkgoldenrod", None, None, None),
    ("RBF.txt", "silver", "dimgray", "RBF+UF", 7.4, 8e-12),
    ("UF.txt", "silver", "dimgray", None, None, None),
]
# (g [GeV^-1], fill colour, label, label x [ueV], label va). The two astro
# bounds are only 1.2x apart, so they are drawn as stacked filled regions
# -- CAST above its value, HB/AGB in the thin strip below it -- exactly as
# in Ref. Ho2026Manu3's Fig. 4, rather than as two nearly coincident lines.
ASTRO_BOUNDS = [
    (5.67e-11, "darkturquoise", "CAST helioscope", 0.115, "bottom"),
    (4.7e-11, "magenta", "HB/AGB stars (globular clusters)", 0.115, "top"),
]
YLIM = (1e-17, 5e-10)

# m_a c^2 = h f, with h = 4.135667696e-15 eV s: f[Hz] = m_a[ueV] * this
HZ_PER_UEV = 1e-6 / 4.135667696e-15


def blend(color, alpha):
    """Pre-blend `color` with white at `alpha`.

    EPS has no alpha channel -- matplotlib's PostScript backend renders a
    semi-transparent patch fully opaque, which would paint a solid block
    over the figure. Mixing the colour with the white background here gives
    exactly the on-screen appearance in a format that has no transparency.
    """
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(color)
    return (alpha * r + (1 - alpha), alpha * g + (1 - alpha), alpha * b + (1 - alpha))


def load_step_envelope(fname):
    """Load an AxionLimits polygon-style limit file and collapse it to the
    lower (most constraining) envelope. Rows at g >= 1 are top-of-plot
    polygon-closure brackets and are dropped. Returns (mass [ueV], g)."""
    m, g = [], []
    with open(fname) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            mm, gg = float(parts[0]), float(parts[1])
            if gg >= 1.0:
                continue
            m.append(mm * 1e6)          # eV -> ueV, this figure's x unit
            g.append(gg)
    m, g = np.array(m), np.array(g)
    order = np.argsort(m)
    return m[order], g[order]

if __name__ == "__main__":
    ps.use_paper_style()

    # one file per N (run_paper_fig3_largeN.py); fall back to the older
    # single-file dense result if no large-N files are present.
    per_N = sorted(glob.glob(data_path("paper_fig3_largeN_N*.json")),
                   key=lambda f: int(re.search(r"_N(\d+)\.json$", f).group(1)))
    curves = {}
    if per_N:
        label = "collective closed form (large N)"
        for f_ in per_N:
            with open(f_) as fh:
                d = json.load(fh)
            curves[int(d["N"])] = d["points"]
            res = d          # shared metadata: masses, benchmarks, exposure
        print(f"Using {len(per_N)} per-N files [{label}]: N = "
              + ", ".join(str(n) for n in sorted(curves)))
    elif os.path.exists(data_path("fig4_reach.json")):
        label = "dense 2^N path"
        with open(data_path("fig4_reach.json")) as fh:
            res = json.load(fh)
        for k, block in res["points"].items():
            curves[int(k)] = block.get("ensemble") or block["full"]
        print(f"Using fig4_reach.json [{label}]")
    else:
        sys.exit("No sensitivity data found -- run run_paper_fig3_largeN.py first.")

    masses = np.array(res["masses_ueV"], dtype=float)
    g_ksvz = np.array(res["g_KSVZ"], dtype=float)
    g_dfsz = np.array(res["g_DFSZ"], dtype=float)
    n_try_scan = res["N_TRY_SCAN"]
    N_list = sorted(curves) if N_SHOW is None else [n for n in sorted(curves) if n in N_SHOW]
    if not N_list:
        sys.exit(f"N_SHOW={N_SHOW} matches none of the available N: {sorted(curves)}")
    colors = ps.n_ramp(len(N_list))

    # Canvas 3.40 x 2.51 in, i.e. w:h = 1.356 -- the proportions of
    # Ref. Ho2026Manu3's Fig. 4. Margins are explicit (and the figure is
    # saved with tight=False) so the saved canvas is exactly that ratio;
    # bbox_inches="tight" would crop to the ink and change it.
    fig, (ax,) = ps.fig_wide(
        1, width=ps.COL_W, height=ps.COL_W / 1.3564,
        margins=dict(left=0.62, right=0.10, bottom=0.40, top=0.24, gap=0.0))

    # ---- existing constraints first, so they sit behind the curves ----
    for fname, fill_c, edge_c, lab, tx, ty in EXPERIMENTS:
        path = data_path(os.path.join("bounds", fname))
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        mb, gb = load_step_envelope(path)
        ax.fill_between(mb, gb, 1.0, color=blend(fill_c, 0.55), lw=0, zorder=1)
        ax.plot(mb, gb, color=edge_c, lw=0.7, zorder=2)
        if lab:
            ax.text(tx, ty, lab, fontsize=6.5, color=edge_c, ha="center",
                    rotation=90, va="center", zorder=4)
    g_top = YLIM[1]
    for i_a, (g_b, c_, lab, tx, va) in enumerate(ASTRO_BOUNDS):
        upper = g_top if i_a == 0 else ASTRO_BOUNDS[i_a - 1][0]
        ax.fill_between([masses[0], masses[-1]], g_b, upper,
                        color=blend(c_, 0.55), lw=0, zorder=1)
        ax.axhline(g_b, color=c_, lw=0.8, zorder=2)
        y_lab = g_b * (2.2 if va == "bottom" else 0.55)
        ax.text(tx, y_lab, lab, fontsize=6, color="0.15", ha="left",
                va=va, zorder=4)

    # benchmark models: dotted KSVZ / dash-dot DFSZ, as in Ho2026Manu3
    ax.plot(masses, g_ksvz, ":", color="0.1", lw=1.5, label="KSVZ", zorder=3)
    ax.plot(masses, g_dfsz, "-.", color="0.1", lw=1.5, label="DFSZ", zorder=3)

    for i, N in enumerate(N_list):
        # trajectory-resolved and trajectory-averaged QFI coincide at this
        # signal strength, so a single curve per N is plotted.
        rows = curves[N]
        sig = np.array([r["sigma_g"] for r in rows], dtype=float)
        m = np.array([r["m_a_ueV"] for r in rows], dtype=float)
        ax.plot(m, sig, marker=ps.n_marker(i), ls=ps.n_linestyle(i), color=colors[i],
                label=rf"$N={N}$", mfc="none", mew=1.2, zorder=5)
        print(f"  N={N:<4} {sig[0]/g_ksvz[0]:7.1f}x -> {sig[-1]/g_ksvz[-1]:6.1f}x KSVZ")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$m_a$  [$\mu$eV]")
    ax.set_ylabel(r"$\sigma_g$  [GeV$^{-1}$]")
    ax.set_xlim(masses[0], masses[-1])
    ax.set_ylim(*YLIM)
    # two legends, as in Ho2026Manu3's Fig. 4: the benchmark models on their
    # own, and the network sizes separately. The band between the
    # astrophysical bounds and the sensitivity curves is empty, so both go
    # there rather than on top of the data.
    handles, labels = ax.get_legend_handles_labels()
    leg_bench = ax.legend(handles[:2], labels[:2], loc="upper left",
                          fontsize=6.5, handlelength=2.0, handletextpad=0.5,
                          labelspacing=0.25, bbox_to_anchor=(0.015, 0.755))
    ax.add_artist(leg_bench)
    ax.legend(handles[2:], labels[2:], loc="upper left", ncol=2, fontsize=6.5,
              columnspacing=0.9, handletextpad=0.4, labelspacing=0.25,
              bbox_to_anchor=(0.30, 0.755))

    # top axis: the Compton frequency the mass corresponds to
    secax = ax.secondary_xaxis(
        "top", functions=(lambda m: m * HZ_PER_UEV, lambda f: f / HZ_PER_UEV))
    secax.set_xscale("log")
    # pull the tick numbers and the label in tight: this axis is a
    # secondary readout, not worth a tall header band
    secax.tick_params(pad=0.5, length=2)
    secax.set_xlabel(r"$f$  [Hz]", labelpad=-2)

    # margins are set explicitly by ps.fig_wide -- no tight_layout here
    ps.save(fig, figs_path("paper_fig3_sensitivity_vs_mass"), tight=False)

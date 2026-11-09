# -*- coding: utf-8 -*-
"""
source/paper_style.py
=======================
Shared plotting style for the FOUR paper figures (the `plot_paper_fig*.py`
scripts in ../run/). Nothing here computes physics; it only fixes colors,
markers and labels so all four figures read as one set.

Two encodings are used, and they are deliberately different:

  * ESTIMATOR identity (trajectory-resolved QFI / trajectory-averaged QFI /
    fixed-measurement CFI) is CATEGORICAL -- three unordered things. It gets
    three fixed hues from the Okabe-Ito colorblind-safe palette, each paired
    with its own marker AND its own line style, so the three curves are
    distinguishable in grayscale print and under red-green color blindness,
    not by hue alone. (The previous tab:blue/tab:red/tab:green trio is
    essentially indistinguishable for deuteranopic readers: the red-green
    pair separates by only dE~4, against dE~11 for the palette used here.)

  * SERIES INDEXED BY N (or by gamma*T) use DISTINCT HUES from the same
    Okabe-Ito palette, each with its own marker and line style. A
    sequential single-hue ramp was used here before; it encoded the
    ordering but made the pale end faint and hard to read, and adjacent
    curves were nearly the same colour. Distinct hues are legible at
    column width, and the ordering is carried by the legend, which is
    ordered anyway.

    The palette below deliberately starts with the three estimator hues,
    so no figure ever shows two different meanings in the same colour:
    the estimator trio and the N-indexed series never appear in the same
    panel.

Author: prepared for Le Bin Ho (Tohoku University).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# --- estimator identity: Okabe-Ito, colorblind-safe -----------------
C_TR = "#0072B2"      # blue        -- trajectory-resolved QFI
C_ENS = "#D55E00"     # vermillion  -- trajectory-averaged (accessible) QFI
C_CFI = "#009E73"     # bluish green-- fixed-measurement CFI

M_TR, M_ENS, M_CFI = "o", "s", "^"
LS_TR, LS_ENS, LS_CFI = "-", "--", ":"

# Legend labels: the SYMBOL ALONE, in the notation used in the manuscript
# text. The symbols are defined in Sec. II D and repeated in every caption,
# so spelling out "(trajectory-resolved QFI)" in the legend only makes the
# box wide enough to crowd the data.
L_TR_g = r"$Q_g^{\rm tr}$"
L_ENS_g = r"$Q_g^{\rm ens}$"
L_CFI_g = r"$F_g$"

L_TR_d = r"$F_\delta^{\rm tr}$"
L_ENS_d = r"$F_\delta^{\rm ens}$"
L_CFI_d = r"$F_\delta^{\rm C}$"

ESTIMATORS_g = [("full", C_TR, M_TR, LS_TR, L_TR_g),
                ("ensemble", C_ENS, M_ENS, LS_ENS, L_ENS_g),
                ("cfi_fixed", C_CFI, M_CFI, LS_CFI, L_CFI_g)]

ESTIMATORS_d = [("full", C_TR, M_TR, LS_TR, L_TR_d),
                ("ensemble", C_ENS, M_ENS, LS_ENS, L_ENS_d),
                ("cfi_fixed", C_CFI, M_CFI, LS_CFI, L_CFI_d)]

# --- series indexed by N or gamma*T: distinct hues -------------------
# Okabe-Ito qualitative palette. Every pair separates by dE > 10 for
# normal, deuteranopic and protanopic vision, and every entry is dark
# enough to read as a 1.2 pt line at column width -- which the pale end
# of a sequential ramp was not.
QUAL = [
    "#0072B2",   # blue
    "#D55E00",   # vermillion
    "#009E73",   # bluish green
    "#CC79A7",   # reddish purple
    "#E69F00",   # orange
    "#56B4E9",   # sky blue
    "#8B6F00",   # dark gold
    "#4B0092",   # violet
]
_N_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_N_LINESTYLES = ["-", "--", ":", "-.",
                 (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)),
                 (0, (1, 1)), (0, (7, 2))]


def n_ramp(n_series, **_ignored):
    """Distinct colors for `n_series` curves indexed by N (or gamma*T).

    Kept under the old name so the plot scripts need no change. It no
    longer returns a light-to-dark ramp: see the module docstring.
    """
    return [QUAL[i % len(QUAL)] for i in range(n_series)]


def n_marker(i):
    return _N_MARKERS[i % len(_N_MARKERS)]


def n_linestyle(i):
    """A distinct dash pattern per series, so the curves stay separable
    in grayscale print and for readers who cannot use hue."""
    return _N_LINESTYLES[i % len(_N_LINESTYLES)]


# --- figure geometry ------------------------------------------------
# Every figure in the manuscript sits in ONE COLUMN of the two-column
# revtex4-2 layout, where \columnwidth = 246 pt = 3.40 in. Figures are
# therefore authored AT that width, so LaTeX scales them by 1.0 and a
# label set at 8 pt here prints at 8 pt next to 10 pt body text. Drawing
# a 9-inch-wide figure and letting \includegraphics shrink it to the
# column is what made the labels unreadable: the scale factor was 0.37,
# so 9.5 pt text printed at 3.5 pt.
COL_W = 3.40          # inches; revtex4-2 aps/prd two-column \\columnwidth
PAGE_W = 7.00         # inches; revtex4-2 aps/prd reprint \\textwidth (510 pt)

# Total figure heights, chosen so the finished figure is not a tall strip.
# A true 4:3 overall (2.55 in tall) is not reachable with two stacked
# panels -- ticks, axis label and margins alone take ~0.8 in per panel, so
# each would be left with ~0.5 in of data. These are the closest workable
# shapes: roughly square for two panels, 4:3 for one.
H_ONE = 2.55          # single panel -> 3.40 x 2.55 = 4:3
H_TWO_SHAREX = 3.35   # two panels sharing one x axis -> ~1.06:1
H_TWO = 4.00          # two panels with independent x axes -> ~1:1.13


def use_paper_style(scale=1.0):
    """Defaults for figures drawn at their FINAL printed size (one column).
    Sizes are in points and are what actually appears on the page.

    `scale` multiplies every type size, line width and marker size. Use it
    for a figure authored WIDER than one column: a page-wide figure is
    7.00 in against 3.40 in, so 8 pt type that looks right in a column
    looks undersized across the page even though it prints at 8 pt either
    way. Fig. 6 uses scale=1.4 (11 pt labels, 10 pt ticks). It is NOT a way
    to compensate for includegraphics scaling -- every figure here is
    still authored at the width it is printed at.
    """
    k = float(scale)
    mpl.rcParams.update({
        "font.size": 8 * k,
        "axes.labelsize": 8 * k,
        "axes.titlesize": 8 * k,
        "legend.fontsize": 6.5 * k,
        "xtick.labelsize": 7 * k,
        "ytick.labelsize": 7 * k,
        "axes.linewidth": 0.6 * k,
        "xtick.major.width": 0.6 * k,
        "ytick.major.width": 0.6 * k,
        "xtick.major.size": 2.5 * k,
        "ytick.major.size": 2.5 * k,
        "lines.linewidth": 1.2 * k,
        "lines.markersize": 5.0 * k,
        # a legend box that is invisible over empty space but masks any
        # curve passing behind it -- matters in Fig. 4, where the legend
        # sits over the KSVZ/DFSZ benchmark lines.
        "legend.frameon": True,
        "legend.framealpha": 0.82,
        "legend.edgecolor": "none",
        "legend.borderpad": 0.25,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        "legend.columnspacing": 1.0,
        # a light grid on every panel, drawn BEHIND the data
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "0.85",
        "grid.linewidth": 0.4 * k,
        "grid.alpha": 1.0,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
    })


def fig_col(nrows=1, sharex=False, height=None):
    """A column-width figure with `nrows` panels stacked VERTICALLY.

    Two panels side by side at column width would be 1.7 in each, too
    narrow for axis labels, so multi-panel figures stack. To keep the
    stack from becoming a long strip, panels carry no titles (the caption
    already states the parameters) and, where both panels share the same
    abscissa, pass sharex=True so only the bottom one spends height on
    tick labels and an axis label.
    """
    if height is None:
        if nrows == 1:
            height = H_ONE
        elif nrows == 2:
            height = H_TWO_SHAREX if sharex else H_TWO
        else:
            height = 1.6 * nrows
    fig, axes = plt.subplots(nrows, 1, figsize=(COL_W, height), sharex=sharex,
                             gridspec_kw={"hspace": 0.12 if sharex else 0.42})
    if nrows == 1:
        axes = [axes]
    return fig, list(axes)


def fig_wide(ncols=2, panel_aspect=4.0 / 3.0, height=None, width=None,
             margins=None):
    """A figure with `ncols` panels SIDE BY SIDE.

    `width=None` uses COL_W, i.e. a single-column figure -- the manuscript
    loads it with the ordinary `figure` environment and `\\columnwidth`, so
    it prints at scale 1.0 and an 8 pt label prints at 8 pt. Pass
    `width=PAGE_W` for a page-wide `figure*` instead.

    At COL_W each of two panels is only ~1.2 in wide, so there is no room
    for a legend inside either one: put a single shared legend above the
    row with `fig.legend(...)` and annotate any remaining curves in place.

    `panel_aspect` is the axes-box height/width; 4/3 gives panels 3 wide to
    4 tall. Margins are set in INCHES so the saved canvas is exactly
    `width` -- save with `save(..., tight=False)`.
    """
    if width is None:
        width = COL_W
    if margins is None:
        wide = width > 1.5 * COL_W
        k = mpl.rcParams["font.size"] / 8.0      # follow use_paper_style(scale)
        margins = dict(left=0.62 * k, right=0.10, bottom=0.50 * k,
                       top=0.10, gap=0.62 * k) if wide \
            else dict(left=0.54, right=0.04, bottom=0.44, top=0.26, gap=0.64)
    m = margins
    w_ax = (width - m["left"] - m["right"] - (ncols - 1) * m["gap"]) / ncols
    h_ax = w_ax * panel_aspect
    if height is None:
        height = h_ax + m["bottom"] + m["top"]
    fig, axes = plt.subplots(1, ncols, figsize=(width, height))
    fig.subplots_adjust(left=m["left"] / width, right=1.0 - m["right"] / width,
                        bottom=m["bottom"] / height, top=1.0 - m["top"] / height,
                        wspace=m["gap"] / w_ax)
    if ncols == 1:
        axes = [axes]
    return fig, list(np.atleast_1d(axes))


def panel_label(ax, text, loc="upper left"):
    """(a)/(b) tag INSIDE the axes. `loc` picks the corner, so a panel whose
    top left is taken by a legend or by the data can move it. The white
    backing keeps it legible over a grid line or a curve."""
    x, y, ha, va = {
        "upper left": (0.030, 0.955, "left", "top"),
        "upper right": (0.970, 0.955, "right", "top"),
        "lower left": (0.030, 0.045, "left", "bottom"),
        "lower right": (0.970, 0.045, "right", "bottom"),
    }[loc]
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va=va, ha=ha, zorder=6,
            bbox=dict(boxstyle="square,pad=0.18", fc="white", ec="none", alpha=0.85))


def sci(x, digits=2):
    """LaTeX scientific notation, e.g. 3.16\\times10^{11} -- avoids the
    unreadable '3.16e+11' that Python's default formatting produces."""
    s = f"{x:.{digits}e}"
    mant, exp = s.split("e")
    return rf"{mant}\times10^{{{int(exp)}}}"


def save(fig, out_stem, tight=True, **_ignored):
    """Save the figure as EPS -- the only format used in this project.

    The manuscript is vector-only: every figure goes into main.tex as .eps.
    No PNG is written. To look at a figure, convert a copy outside the
    project tree, e.g.

        gs -dSAFER -dEPSCrop -r200 -sDEVICE=png16m -o /tmp/f.png fig.eps

    Note that PostScript has no alpha channel: partially transparent
    artists (legend frames, the white box behind a panel tag) are
    flattened to fully opaque, which is what we want in print anyway.
    """
    if tight:
        fig.savefig(out_stem + ".eps", bbox_inches="tight")
    else:
        # savefig(bbox_inches=None) falls back to rcParams["savefig.bbox"],
        # which is "tight" here -- override it so the full canvas is written
        with mpl.rc_context({"savefig.bbox": None}):
            fig.savefig(out_stem + ".eps")
    print(f"Saved {out_stem}.eps")

# -*- coding: utf-8 -*-
"""
source/paths.py
=================
Tiny shared helper so every run_fig*.py / plot_fig*.py script agrees on
where code/data/ and code/figs/ are, regardless of the working directory
it's launched from.

Layout (see code/README.md for the full picture):
    code/source/   -- shared library (this package)
    code/run/      -- run_fig*.py (compute, save JSON to code/data/)
                       plot_fig*.py (load JSON, save PNG to code/figs/)
    code/data/     -- computed results, one JSON per figure
    code/figs/     -- plotted figures, one PNG per figure

This is a SEPARATE, parallel sandbox from the original
code/*.py + figures/*.png pipeline that produced the actual figures in
main.tex -- those files are untouched. Use this source/run/data/figs
structure for your own N-scaling exploration.
"""

import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))   # .../code/source
CODE_DIR = os.path.dirname(_THIS_DIR)                     # .../code
DATA_DIR = os.path.join(CODE_DIR, "data")
FIGS_DIR = os.path.join(CODE_DIR, "figs")


def data_path(filename):
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, filename)


def figs_path(filename):
    os.makedirs(FIGS_DIR, exist_ok=True)
    return os.path.join(FIGS_DIR, filename)

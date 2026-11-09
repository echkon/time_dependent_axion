# time_dependent_axion

Code for **"Time-dependent quantum sensing of stochastic axion dark matter
with entangled qubit sensors"** (Le Bin Ho).

The Galactic axion field is a stochastic process: its amplitude and phase
fluctuate on the coherence time `tau_DM`. This repository implements the
time-dependent quantum-metrology framework used in the paper and reproduces
every computed figure. It distinguishes three quantities,

| symbol | meaning |
| --- | --- |
| `Q_g^tr` | trajectory-resolved QFI, averaged over field realizations |
| `Q_g^ens` | QFI of the ensemble-averaged state |
| `F_g` | classical Fisher information of the fixed measurement |

and tracks the projected sensitivity to the axion-photon coupling
`g_agammagamma`, including detuning and independent per-qubit dephasing.

## Layout

```
source/   shared library
run/      run_*.py   compute, write JSON to data/
          plot_*.py  read JSON, write EPS to figs/
data/     computed results, one JSON per figure (included)
figs/     figures produced by the plot scripts
```

`source/paths.py` resolves `data/` and `figs/` relative to the package, so
the scripts work from any working directory.

### Modules

| file | contents |
| --- | --- |
| `source/axion_field.py` | Standard Halo Model field envelope (sum of plane waves), KSVZ/DFSZ benchmark couplings, coupling-to-rate conversion |
| `source/pang_jordan.py` | time-dependent generator and QFI for a known trajectory |
| `source/collective.py` | collective-spin evolution of the N-qubit probe |
| `source/dephasing.py` | closed form and dense reference for independent per-qubit dephasing |
| `source/detuning.py` | QFI under a qubit-field detuning |
| `source/samplers.py`, `source/qubit_ops.py` | field samplers and spin operators |
| `source/paper_style.py`, `source/paths.py` | figure style and path helpers |

## Requirements

Python 3.10 or newer.

```
pip install -r requirements.txt
```

## Reproducing the figures

The JSON results are included, so the plot scripts run immediately:

```
python run/plot_paper_fig1_time_resolved.py
python run/plot_paper_fig2_trajectory_information.py
python run/plot_paper_fig3_sensitivity_vs_mass.py
python run/plot_paper_fig4_detuning.py
python run/plot_paper_fig5_dephasing.py
```

To recompute from scratch, run the corresponding `run_*.py` first. The
compute step is the expensive one; the large-N sweep of Fig. 4 was run on a
cluster.

| figure in the paper | plot script | compute script |
| --- | --- | --- |
| Fig. 2, information and sensitivity vs time | `plot_paper_fig1_time_resolved.py` | `run_fig2_time_resolved.py` |
| Fig. 3, cost `R_g(T)` vs rotation angle | `plot_paper_fig2_trajectory_information.py` | `run_fig3_crossover.py` |
| Fig. 4, sensitivity vs axion mass | `plot_paper_fig3_sensitivity_vs_mass.py` | `run_fig4_reach.py`, `run_paper_fig3_largeN.py` |
| Fig. 5, detuning | `plot_paper_fig4_detuning.py` | `run_fig5_detuning_Nscan.py` |
| Fig. 6, dephasing | `plot_paper_fig5_dephasing.py` | `run_paper_fig5_dephasing.py` |

Figure 1 of the paper is a schematic and is not produced here.

## Checks

| script | what it verifies |
| --- | --- |
| `run/run_weakexp_check.py` | the weak-coupling expansion of the ensemble QFI against direct evaluation |
| `run/run_deph_validation.py`, `run/plot_paper_fig6_deph_validation.py` | the dephasing closed form against dense density-matrix propagation at small N |

## Data from other experiments

`data/bounds/` holds the published ADMX and RBF/UF exclusion limits used as
shaded regions in the sensitivity figure. See `data/bounds/README.md` for
their sources.

## License

MIT, see `LICENSE`.

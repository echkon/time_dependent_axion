# Existing constraints used in Fig. 4

Copied verbatim from the Manu3 project
(`Majoron/Codes/code_noise/topo/data/`), which took them from AxionLimits.
Columns: axion mass [eV], coupling g_agg [GeV^-1]. Rows with g >= 1 are
polygon-closure brackets and are dropped when the lower (most constraining)
envelope is extracted -- see `load_step_envelope()` in
`run/plot_paper_fig3_sensitivity_vs_mass.py`.

| file | experiment | mass range covered |
|---|---|---|
| ADMX2018.txt | ADMX, Phys. Rev. Lett. 120, 151301 (2018) | 2.67-2.79 ueV |
| ADMX2021.txt | ADMX, Phys. Rev. Lett. 127, 261803 (2021) | 3.31-4.21 ueV |
| RBF.txt      | RBF, Phys. Rev. Lett. 59, 839 (1987)      | 4.51-10.2 ueV |
| UF.txt       | UF,  Phys. Rev. D 42, 1297 (1990)         | 5.62-8.06 ueV |

The two astrophysical bounds are single horizontal values, not files:

| bound | g_agg [GeV^-1] | reference |
|---|---|---|
| CAST helioscope            | 5.67e-11 | Anastassopoulos et al. (2017) |
| HB/AGB stars (glob. clus.) | 4.7e-11  | Dolan et al. (2022)           |

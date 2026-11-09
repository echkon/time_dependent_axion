# -*- coding: utf-8 -*-
"""
source/pang_jordan.py
=======================
The core, model-independent time-dependent-Hamiltonian Fisher-information
machinery used by EVERY figure in this project (Pang & Jordan, Nat.
Commun. 8, 14695 (2017), generator formalism). This module knows nothing
about phase diffusion or the axion field specifically -- it only needs a
"sampler" function that returns one realization of the complex drive
trajectory z(t) = x(t) + i*y(t) on a uniform time grid, where the
single-qubit encoding Hamiltonian is

    H_1(t; s) = hbar * s * ( Re[z(t)] sx + Im[z(t)] sy )

and s is the (scalar) parameter to estimate (either the toy-model
rotation delta, or the physically-derived rate s(g_agg) -- see
`source/axion_field.py` for the g_agg <-> s conversion). Every
figure-specific run_fig*.py script only has to supply the right sampler:
  - Wiener toy model: `source/samplers.py`'s `wiener_sampler` returns
    z(t) = exp(i*alpha(t)), alpha(t) a Wiener phase-diffusion path.
  - Physically-grounded field: `source/axion_field.py`'s
    `sample_axion_envelope` returns z(t) = zeta(t) directly (the
    normalized sum-of-modes DM field envelope, |zeta| itself
    fluctuating -- NOT unit modulus, unlike the Wiener case).
  - Detuning: `source/samplers.py`'s `make_detuned_sampler` wraps any
    existing sampler with a deterministic e^{-i*Delta*t} phase ramp.

For i*hbar*dU/dt = H(t)U(t), U(0)=I, the EXACT (not perturbative)
derivative of the propagator w.r.t. s is (variation of parameters):
    d_s U(T) = -i U(T) * H_m(T),
    H_m(T) = Integral_0^T U(t)^dagger [d_s H(t)] U(t) dt   (Hermitian).
For a SINGLE trajectory (pure state), the s-QFI is exactly
F = 4*Var_psi0(H_m(T)) -- no approximation, valid for any s, not just
small s (no finite differences are used anywhere in this codebase).

Two estimation strategies:
  - FULL-INFORMATION (perfect real-time tracking): evaluate H_m(T) with
    the TRUE realized trajectory for each Monte Carlo shot, then average
    F_traj = 4*Var_psi0(H_m(T)) over the ensemble.
  - ENSEMBLE (realistic end-of-shot readout, trajectory unknown): build
    the ACTUAL physical (mixed) state rho_bar(s) = E_traj[|psi_T><psi_T|]
    and its exact derivative d_s rho_bar, then apply the standard
    mixed-state (SLD) QFI formula.
A third, deliberately non-optimized estimator is also available: the
classical Fisher information (CFI) of a FIXED measurement in the
computational (Z) basis -- see `time_resolved_ensemble`'s docstring.

Author: prepared for Le Bin Ho (Tohoku University) -- Manus/Manu4.
"""

import numpy as np
from scipy.linalg import expm

from .qubit_ops import sx, sy, I2, collective_sum, ghzX_state


def sld_qfi_single_param(rho, drho, eps=1e-12):
    """Standard mixed-state (SLD, eigenbasis) single-parameter QFI:
    F = 2 * sum_{i,j: w_i+w_j>eps} |<i|drho|j>|^2 / (w_i+w_j), {w,v} the
    eigendecomposition of rho. Same algorithm used by Manu3's
    onecircuit.metric.sld_qfim_eigens, reimplemented here in plain numpy."""
    w, v = np.linalg.eigh(rho)
    d = rho.shape[0]
    F = 0.0
    for i in range(d):
        for j in range(d):
            denom = w[i] + w[j]
            if denom > eps:
                num = (v[:, i].conj() @ drho @ v[:, j]) * (v[:, j].conj() @ drho @ v[:, i])
                F += 2.0 * np.real(num) / denom
    return F


def time_resolved_ensemble(field_sampler, s_scale, T, nsteps, N, n_traj, rng,
                            checkpoint_steps, psi0=None):
    """THE core routine -- generic, model-independent time-resolved
    full-information / ensemble / fixed-basis-CFI computation.

    field_sampler(rng) -> complex array of length nsteps+1, the drive
    direction z(t)=x(t)+i*y(t) on a uniform grid over [0,T]; the
    single-qubit encoding Hamiltonian is H_1(t)=s_scale*(Re[z(t)] sx +
    Im[z(t)] sy). checkpoint_steps: sorted array of step indices
    (1..nsteps) at which to record. Returns a dict of arrays, one entry
    per checkpoint: full/full_err (full-information QFI, mean+-SEM over
    n_traj trajectories), ensemble/cfi_fixed (purity-masked SLD QFI and
    fixed-Z-basis CFI), ensemble_raw/cfi_fixed_raw (unmasked, for
    diagnosing the purity guard below), purity, reliable (bool mask).

    Fixed-basis CFI note: this measures directly in the COMPUTATIONAL (Z)
    basis, NOT the seemingly more natural Hadamard/X basis. The X-GHZ
    probe |psi0>=(|+>^N+|->^N)/sqrt2, written in the Z basis, is an
    equal-amplitude superposition of exactly the EVEN-PARITY bit-strings;
    measuring this parity structure in the Z basis exactly saturates the
    QFI at t=0 (CFI_Z=4N^2T^2 to machine precision), whereas the
    Hadamard/X basis (the basis the |+>,|-> branches are themselves built
    from) gives essentially zero signal -- the standard "a GHZ probe
    needs a disentangling readout" fact of phase-sensitive GHZ metrology.

    Numerical-reliability guard: at very early times / very small true
    signal, rho_bar(t) can be numerically almost EXACTLY pure (1-purity
    at or below the double-precision noise floor), which makes the
    eigendecomposition inside sld_qfi_single_param (and the diagonal CFI
    sum) numerically unreliable. We mask ensemble/cfi_fixed to NaN
    whenever 1-purity <= 1e-12 (the pure-state full-information curve,
    which needs no eigendecomposition, is never affected)."""
    if psi0 is None:
        psi0 = ghzX_state(N)
    dim = 2 ** N
    dt = T / nsteps
    cp_set = set(int(c) for c in checkpoint_steps)
    n_cp = len(checkpoint_steps)
    cp_index = {c: i for i, c in enumerate(sorted(cp_set))}

    full_sum = np.zeros(n_cp)
    full_sumsq = np.zeros(n_cp)
    rho_bar = [np.zeros((dim, dim), dtype=complex) for _ in range(n_cp)]
    drho_bar = [np.zeros((dim, dim), dtype=complex) for _ in range(n_cp)]

    for _ in range(n_traj):
        zt = field_sampler(rng)
        U1 = I2.copy()
        h1 = np.zeros((2, 2), dtype=complex)
        for i in range(nsteps):
            zm = 0.5 * (zt[i] + zt[i + 1])
            dH1 = zm.real * sx + zm.imag * sy
            H = s_scale * dH1
            Ustep = expm(-1j * H * dt)
            h1 = h1 + (U1.conj().T @ dH1 @ U1) * dt
            U1 = Ustep @ U1
            step_num = i + 1
            if step_num in cp_set:
                idx = cp_index[step_num]
                U_N = U1
                for _ in range(N - 1):
                    U_N = np.kron(U_N, U1)
                H_gen = collective_sum(h1, N)
                e1 = np.real(psi0.conj() @ H_gen @ psi0)
                e2 = np.real(psi0.conj() @ (H_gen @ H_gen) @ psi0)
                F = 4.0 * (e2 - e1 ** 2)
                full_sum[idx] += F
                full_sumsq[idx] += F ** 2
                psi_t = U_N @ psi0
                dpsi_t = -1j * (U_N @ (H_gen @ psi0))
                rho_bar[idx] += np.outer(psi_t, psi_t.conj())
                drho_bar[idx] += np.outer(dpsi_t, psi_t.conj()) + np.outer(psi_t, dpsi_t.conj())

    full_mean = full_sum / n_traj
    full_err = np.sqrt(np.maximum(full_sumsq / n_traj - full_mean ** 2, 0.0) / n_traj)

    ensemble = np.empty(n_cp)
    cfi_fixed = np.empty(n_cp)
    purity = np.empty(n_cp)
    eps = 1e-14
    for i in range(n_cp):
        rb = rho_bar[i] / n_traj
        drb = drho_bar[i] / n_traj
        purity[i] = np.real(np.trace(rb @ rb))
        ensemble[i] = sld_qfi_single_param(rb, drb)
        p = np.real(np.diag(rb))
        dp = np.real(np.diag(drb))
        cfi_fixed[i] = np.sum(dp ** 2 / np.maximum(p, eps))

    reliable = (1.0 - purity) > 1e-12
    ensemble_masked = np.where(reliable, ensemble, np.nan)
    cfi_fixed_masked = np.where(reliable, cfi_fixed, np.nan)

    return {"full": full_mean, "full_err": full_err,
            "ensemble": ensemble_masked, "cfi_fixed": cfi_fixed_masked,
            "ensemble_raw": ensemble, "cfi_fixed_raw": cfi_fixed,
            "purity": purity, "reliable": reliable}


def endpoint(field_sampler, s_scale, T, nsteps, N, n_traj, rng, psi0=None):
    """Convenience: run time_resolved_ensemble with a SINGLE checkpoint at
    t=T (the end of the shot) and return the resulting scalar dict --
    this is what every END-OF-SHOT figure (1, 3, 4, 5) actually needs;
    only Fig. 2 (the genuinely time-resolved figure) uses the full
    checkpointed `time_resolved_ensemble` output directly."""
    res = time_resolved_ensemble(field_sampler, s_scale, T, nsteps, N, n_traj, rng,
                                  checkpoint_steps=np.array([nsteps]), psi0=psi0)
    return {k: (v[0] if hasattr(v, "__len__") else v) for k, v in res.items()}


def static_qfi_analytic(T, N, psi0=None):
    """kappa=0, alpha frozen at alpha0=0 (or equivalently z(t)=1
    identically): H_m(T) = T*sigma_x exactly (no time integration
    needed). Cross-checks `endpoint`/`time_resolved_ensemble` at
    kappa=0 -- Var(sum sigma_x) = N^2 exactly for the X-GHZ probe."""
    if psi0 is None:
        psi0 = ghzX_state(N)
    H_gen = T * collective_sum(sx, N)
    e1 = np.real(psi0.conj() @ H_gen @ psi0)
    e2 = np.real(psi0.conj() @ (H_gen @ H_gen) @ psi0)
    return 4.0 * (e2 - e1 ** 2)

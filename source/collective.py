# -*- coding: utf-8 -*-
"""
source/collective.py
======================
LARGE-N evaluation of the trajectory-resolved QFI, with a cost that is
INDEPENDENT of the qubit number N.

Why this is possible. Every qubit of the network sees the same field, so
the collective Pang-Jordan generator is a sum of identical single-qubit
generators,

    H_N(T) = sum_k h1^(k)(T),      h1(T) = c0 I + c_x sx + c_y sy + c_z sz,

and h1 is a 2x2 matrix obtained by integrating a 2x2 Schrodinger equation
-- that part never knew about N in the first place. Writing
S_a = sum_k sigma_a^(k), the trajectory-resolved QFI is

    F = 4 Var_psi0[H_N(T)] = 4 sum_{a,b} c_a c_b M_ab(N),

so all the N-dependence sits in the moments M_ab of the collective spin
operators on the probe state. For the X-GHZ probe
|psi0> = (|+>^N + |->^N)/sqrt(2) those moments are exactly

    <S_x> = <S_y> = <S_z> = 0,     M = diag(N^2, N, N)      (N >= 3),

with two special cases, N = 1 (<S_z> = 1, M = diag(1, 1, 0)) and N = 2
(M = diag(4, 0, 4)). Verified against the dense 2^N construction for
N = 1..10 to machine precision. Hence

    F = 4 [ c_x^2 N^2 + (c_y^2 + c_z^2) N ]          (N >= 3),

the N^2 Heisenberg term appearing only along the probe's fiducial sigma_x
axis -- the same anisotropy that Appendix C's detuning formula exposes.

The practical consequence: the dense path in `pang_jordan.py` allocates
2^N x 2^N density matrices and stops being feasible around N = 13-14
(N = 14 already needs ~4 GB per matrix, and two are required), whereas
the routine here never builds anything larger than 2x2 and runs at
N = 20, 50 or 1000 in the same time it takes at N = 5.

WHAT THIS DOES NOT GIVE YOU. Only the TRAJECTORY-RESOLVED quantity is
collective in this way. The trajectory-averaged QFI needs the mixed state
rho_bar = E[|psi_T><psi_T|] and the fixed-measurement CFI needs its
diagonal, and neither survives the average as a 2x2 object -- those stay
on the dense path and stay capped near N = 13. For the coupling-
sensitivity figure this is not a limitation, because at the physical
KSVZ signal strength the two QFIs agree to ~1e-4 % (Sec. III A/III B), so
the trajectory-resolved value is the sensitivity; for any figure that
plots the two against each other, the dense path is still required.

Author: prepared for Le Bin Ho (Tohoku University).
"""

import numpy as np
from scipy.linalg import expm

from .qubit_ops import sx, sy, sz, I2


def ghzX_moments(N):
    """Exact (<S>, M) for the X-GHZ probe: the first moments as a length-3
    vector and the covariance M_ab = <{S_a,S_b}>/2 - <S_a><S_b> as a 3x3
    matrix, in the order (x, y, z)."""
    if N == 1:
        return np.array([0.0, 0.0, 1.0]), np.diag([1.0, 1.0, 0.0])
    if N == 2:
        return np.zeros(3), np.diag([4.0, 0.0, 4.0])
    return np.zeros(3), np.diag([float(N) ** 2, float(N), float(N)])


def qfi_from_generator(h1, N):
    """Trajectory-resolved QFI 4*Var_psi0[sum_k h1^(k)] for the X-GHZ probe,
    from the single-qubit 2x2 generator h1 alone."""
    c = np.array([0.5 * np.real(np.trace(h1 @ op)) for op in (sx, sy, sz)])
    mean, M = ghzX_moments(N)
    var = float(c @ M @ c)
    return 4.0 * var


def endpoint_tr_collective(field_sampler, s_scale, T, nsteps, N, n_traj, rng):
    """Trajectory-resolved QFI at t = T, Monte Carlo averaged, using only
    2x2 operations. Drop-in replacement for `pang_jordan.endpoint`'s
    "full"/"full_err" entries; returns the same two numbers in a dict.

    Cost is O(n_traj * nsteps) and does NOT depend on N.
    """
    dt = T / nsteps
    acc = np.empty(n_traj)
    for j in range(n_traj):
        zt = field_sampler(rng)
        U1 = I2.copy()
        h1 = np.zeros((2, 2), dtype=complex)
        for i in range(nsteps):
            zm = 0.5 * (zt[i] + zt[i + 1])
            dH1 = zm.real * sx + zm.imag * sy
            H = s_scale * dH1
            h1 = h1 + (U1.conj().T @ dH1 @ U1) * dt
            U1 = expm(-1j * H * dt) @ U1
        acc[j] = qfi_from_generator(h1, N)
    mean = float(np.mean(acc))
    return {"full": mean, "full_err": float(np.std(acc) / np.sqrt(n_traj))}

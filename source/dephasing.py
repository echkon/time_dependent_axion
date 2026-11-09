# -*- coding: utf-8 -*-
"""
source/dephasing.py
=====================
Independent per-qubit dephasing during the interrogation, and what it does
to the collective enhancement of the X-GHZ probe.

MODEL. Each qubit dephases independently at rate gamma about a chosen axis
while the field encoding proceeds. In Kraus form, per qubit per step,
    rho -> (1-p) rho + p A rho A ,   A = sigma_axis,
    p = [1 - exp(-2 gamma dt)]/2 ,
so a single-qubit coherence transverse to that axis decays as
exp(-2 gamma t). Only the dimensionless product gamma*T matters.

WHAT WE FOUND (all verified against the exact dense simulation below).
The answer depends on the axis, and the reason is the probe's anisotropy --
the same anisotropy that gives the N^2 enhancement along sigma_x only
(Appendix B.1) and that shapes the detuning response (Appendix C).

  * Dephasing ALONG THE FIDUCIAL AXIS (sigma_x) attacks exactly the
    coherence between the two branches |+>^N and |->^N that carries the
    Heisenberg term, and does so N times faster than for a single qubit:

        c_x^2 N^2   ->   c_x^2 N^2 exp(-4 N gamma T) .

    This is EXACT: it reproduces the dense 2^N simulation to ~1e-14
    relative over gamma*T = 0.05..0.5 and N = 2..8.

  * The remaining components are NOT exponentially suppressed in N. They
    live in single-qubit coherences and are damped only at the one-qubit
    rate, exp(-2 gamma T), independent of N.

  * Dephasing along sigma_z is comparatively benign: the loss is weakly
    N-dependent and saturates (e.g. a factor 0.84 at gamma*T = 0.1 for
    N = 8, against exp(-4 N gamma T) = 0.04 for the sigma_x case).

Putting the two surviving pieces together,

    F(N) = 4 [ c_x^2 N^2 exp(-4 N gamma T)
               + (c_y^2 + c_z^2) N exp(-2 gamma T) ] ,

which matches the exact simulation for the physical axion field to better
than 0.5% at gamma*T = 0.02 and 1.4% at gamma*T = 0.05. The first term is
exact; the second is the approximation, and the quoted figures are its
accuracy.

CONSEQUENCE. There is no hard optimal network size. Instead there is a
CROSSOVER at

    N_c ~ 1 / (2 gamma T) ,

where the exponential overtakes the N^2 growth. Below N_c the sensitivity
improves as 1/N (Heisenberg); above it the Heisenberg term is
exponentially dead and only the linear term survives, so the network
reverts to the standard 1/sqrt(N) scaling. Keeping a Heisenberg advantage
at a given N therefore requires gamma*T <~ 1/(2N) -- at N = 100, a
fiducial-axis dephasing time longer than about 200 interrogation times.

Author: prepared for Le Bin Ho (Tohoku University).
"""

import numpy as np
from scipy.linalg import expm

from .qubit_ops import sx, sy, sz, I2, embed, collective_sum, ghzX_state
from .pang_jordan import sld_qfi_single_param
from .collective import ghzX_moments


def generator_coeffs(field_sampler, s_scale, T, nsteps, rng):
    """The 2x2 generator coefficients (c_x, c_y, c_z) for one field
    realization. Cost independent of N."""
    zt = field_sampler(rng)
    dt = T / nsteps
    U1 = I2.copy()
    h1 = np.zeros((2, 2), dtype=complex)
    for i in range(nsteps):
        zm = 0.5 * (zt[i] + zt[i + 1])
        dH1 = zm.real * sx + zm.imag * sy
        h1 = h1 + (U1.conj().T @ dH1 @ U1) * dt
        U1 = expm(-1j * s_scale * dH1 * dt) @ U1
    return np.array([0.5 * np.real(np.trace(h1 @ op)) for op in (sx, sy, sz)])


def qfi_dephased(c, N, gammaT):
    """F(N) under fiducial-axis dephasing, from the coefficients c.
    Exact in its first term, approximate in the second (see module docstring).

    The N-dependence is taken from ghzX_moments(N) rather than from the
    N >= 3 formula, so the two exceptions of Appendix B -- M = diag(1,1,0)
    at N = 1 and M = diag(4,0,4) at N = 2 -- are used as written. Ignoring
    them overestimates F by ~9% at N = 2. M[0,0] carries the branch
    coherence and is damped by exp(-4 N gamma T); the remaining diagonal
    entries live in single-qubit coherences and are damped by
    exp(-2 gamma T).
    """
    _, M = ghzX_moments(N)
    heis = c[0] ** 2 * M[0, 0] * np.exp(-4.0 * N * gammaT)
    rest = (c[1] ** 2 * M[1, 1] + c[2] ** 2 * M[2, 2]) * np.exp(-2.0 * gammaT)
    return 4.0 * (heis + rest)


def crossover_N(gammaT):
    """N_c ~ 1/(2 gamma T): where the exponential overtakes N^2 growth."""
    return np.inf if gammaT <= 0 else 1.0 / (2.0 * gammaT)


def qfi_dephased_dense(zt, s_scale, T, nsteps, N, gammaT, axis="x"):
    """EXACT reference: propagates (rho, d rho/ds) jointly through unitary
    steps and the Kraus dephasing channel, then takes the SLD QFI. Used to
    validate `qfi_dephased`; costs 2^N x 2^N, so N <~ 9."""
    A1 = {"x": sx, "y": sy, "z": sz}[axis]
    dim = 2 ** N
    dt = T / nsteps
    gamma = gammaT / T
    p = 0.5 * (1.0 - np.exp(-2.0 * gamma * dt))
    Aks = [embed(A1, k, N) for k in range(N)]

    psi0 = ghzX_state(N)
    rho = np.outer(psi0, psi0.conj())
    drho = np.zeros((dim, dim), dtype=complex)

    for i in range(nsteps):
        zm = 0.5 * (zt[i] + zt[i + 1])
        dH = collective_sum(zm.real * sx + zm.imag * sy, N)
        U = expm(-1j * s_scale * dH * dt)
        dU = -1j * dt * dH @ U
        drho = dU @ rho @ U.conj().T + U @ drho @ U.conj().T + U @ rho @ dU.conj().T
        rho = U @ rho @ U.conj().T
        if gamma > 0:
            for Ak in Aks:
                rho = (1 - p) * rho + p * (Ak @ rho @ Ak)
                drho = (1 - p) * drho + p * (Ak @ drho @ Ak)
    return sld_qfi_single_param(rho, drho)


# =====================================================================
# ENERGY-BASIS (sigma_z) DEPHASING SET BY T2
# =====================================================================
# The interrogation time is T = min(tau_DM, T2) with T2 = 100 us, so every
# shot in this paper runs for exactly one T2. A transmon's T2 is dephasing
# in the ENERGY basis, i.e. a sigma_z channel, which is NOT the sigma_x
# channel studied above -- sigma_x dephasing is a separate, fiducial-axis
# process (drive-axis/control noise). Both are present; the curves reported
# elsewhere modelled only the sigma_x one, so they must be corrected by the
# sigma_z factor measured here.
#
# With T = T2 the sigma_z rate is fixed: coherence ~ exp(-2 gamma_z t) and
# T2 = 1/(2 gamma_z), hence gamma_z*T = T/(2 T2) = 0.5.
#
# In the |+->/|-> basis the probe is an ordinary GHZ state and sigma_z is a
# BIT FLIP. With n of N qubits flipped the branch phase is 2 c_x (N - 2n)
# with n ~ Binomial(N, q), q = [1 - exp(-2 gamma_z t)]/2, so
#     E[(N-2n)^2] / N^2 = (1-2q)^2 + 4q(1-q)/N ,
# i.e. the QFI ratio has the form a + b/N. That is the structural reason
# for the fit below; a and b themselves are measured, because the static
# argument ignores WHEN during the shot a flip occurs.
#
# MEASURED with qfi_dephased_dense(axis="z"), ratio of ENSEMBLE MEANS over
# 12 field realizations, nsteps = 40, m_a = 1 ueV, gamma_z*T = 0.5:
#     N    1       2       3       4       5       6       7
#     R  0.737   0.913   0.608   0.575   0.551   0.532   0.517
# least squares on N >= 3 gives R(N) = 0.4528 + 0.4731/N, residual < 0.004.
#
# TIME DEPENDENCE, measured the same way at N = 6:
#     t/T   0.10    0.25    0.50    0.75    1.00
#     R    0.9377  0.8274  0.6875  0.6128  0.5356
# used below as a shape only, assumed N-independent.
#
# CAVEAT: R(N) beyond N = 7 is the fit, not a measurement; the dense
# reference is capped near N = 9 by the 2^N cost.
# =====================================================================

_RZ_MEASURED = {1: 0.7368, 2: 0.9131, 3: 0.6076, 4: 0.5746,
                5: 0.5507, 6: 0.5317, 7: 0.5166}
_RZ_FIT_A, _RZ_FIT_B = 0.4528, 0.4731
_RZ_SHAPE_X = np.array([0.0, 0.10, 0.25, 0.50, 0.75, 1.00])
_RZ_SHAPE_F = np.array([0.0, 0.1342, 0.3717, 0.6729, 0.8338, 1.0])


def sigma_z_factor(N, t_over_T=1.0):
    """QFI multiplier from the energy-basis (sigma_z) dephasing implied by
    T = T2, i.e. gamma_z*T = 0.5. Returns R in (0, 1]; the projected
    sensitivity degrades as sigma_g -> sigma_g / sqrt(R)."""
    N = int(N)
    R_end = _RZ_MEASURED.get(N, _RZ_FIT_A + _RZ_FIT_B / N)
    if t_over_T >= 1.0:
        return R_end
    f = float(np.interp(t_over_T, _RZ_SHAPE_X, _RZ_SHAPE_F))
    return 1.0 - (1.0 - R_end) * f


def optimal_N(cx2, cyz2, gammaT, N_max=20000):
    """The network size that actually minimises sigma_g under fiducial-axis
    dephasing, located directly on Eq. (deph) rather than assumed equal to
    N_c = 1/(2 gamma T).

    With A = cx2 * exp(-4 N gamma T) and B = cyz2 * exp(-2 gamma T), the QFI
    is 4(A N^2 + B N) and stationarity requires

        2 A N (1 - 2 N gamma T) + B = 0 ,

    not 1 - 2 N gamma T = 0. Since B > 0 the root sits slightly ABOVE N_c,
    and the stationary point is a LOCAL minimum of sigma_g only: beyond the
    following local maximum the linear term restores 1/sqrt(N) improvement.
    Returns None when no interior minimum exists below N_max.
    """
    if gammaT <= 0:
        return None
    N = np.arange(1, N_max + 1, dtype=float)
    q = np.array([qfi_dephased(np.array([np.sqrt(cx2), np.sqrt(cyz2), 0.0]),
                               int(n), gammaT) for n in N])
    sg = 1.0 / np.sqrt(q)
    for i in range(1, len(sg) - 1):
        if sg[i] < sg[i - 1] and sg[i] < sg[i + 1]:
            return int(N[i])
    return None

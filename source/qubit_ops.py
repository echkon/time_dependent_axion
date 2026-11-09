# -*- coding: utf-8 -*-
"""
source/qubit_ops.py
=====================
Shared N-qubit operator building blocks: Pauli matrices, single-qubit
embedding into an N-qubit register, collective (identical, local)
generators, and the X-basis GHZ probe state used throughout Manu4.

This is pure linear algebra with NO physics-model content (no axion
field, no phase diffusion) -- every run_fig*.py script imports from here.

Cost note (relevant when increasing N): all state vectors/operators here
are DENSE numpy arrays of dimension dim=2**N. Memory for one dim x dim
complex matrix is ~16*dim^2 bytes, so N=10 (dim=1024) is ~16 MB/matrix,
N=13 (dim=8192) is ~1 GB/matrix -- several such matrices exist at once
inside the Pang-Jordan machinery in `pang_jordan.py`, and embed()/
collective_sum() below built via repeated np.kron scale as O(dim^2) time
each. Practically: N up to ~10-11 is comfortable on a laptop; N=12-13 is
slow but usually still feasible (minutes per run); N>=14 (dim>=16384) is
not recommended without switching to a sparse/tensor-network
representation, which this codebase does not implement.

Author: prepared for Le Bin Ho (Tohoku University) -- Manus/Manu4.
"""

import numpy as np

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def embed(op, k, N):
    """Embed a single-qubit operator `op` on qubit k (0-indexed) of an
    N-qubit register; identity elsewhere."""
    ops = [I2] * N
    ops[k] = op
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def collective_sum(op, N):
    """Sum_k embed(op, k, N) -- the collective N-qubit generator for a
    Hamiltonian that couples identically and locally to every qubit."""
    dim = 2 ** N
    out = np.zeros((dim, dim), dtype=complex)
    for k in range(N):
        out += embed(op, k, N)
    return out


def ghzX_state(N):
    """(|+>^N + |->^N)/sqrt(2), the sigma_x collective "cat" state:
    exactly saturates Var(sum_k sigma_x^(k)) = N^2 (Heisenberg scaling
    along the probe's fiducial axis). This is the probe used for every
    figure in this project."""
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
    a, b = plus.copy(), minus.copy()
    for _ in range(N - 1):
        a = np.kron(a, plus)
        b = np.kron(b, minus)
    return (a + b) / np.sqrt(2)

"""
State-space representation of Delta^m Y_k = U_k (Sec. II-B, eq. (7); [20, eq. (39)]).

State X_k = [Y_{k+m-1}, ..., Y_{k+1}, Y_k]^T in R^m,

    A = [[a_{m-1}, a_{m-2}, ..., a_1, a_0],     a_i = (-1)^{i+m-1} binom(m, i),
         [1, 0, ..., 0, 0],
         ...
         [0, ..., 0, 1, 0]]

B = [1, 0, ..., 0]^T (U_k enters the first component), C = [0, ..., 0, 1]
(Y_k is the last component) and B' = [1, ..., 1]^T (the level jump U'_k
shifts all states equally, Sec. II-C).  For m = 2: A = [[2, -1], [1, 0]].
"""
from math import comb

import numpy as np


def build_ssm(m: int):
    """Return (A, B, C, B_prime) as 2-D A and 1-D vectors B, C, B'."""
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    A = np.zeros((m, m))
    A[0, :] = [(-1) ** (i + m - 1) * comb(m, i) for i in range(m - 1, -1, -1)]
    for i in range(1, m):
        A[i, i - 1] = 1.0
    B = np.zeros(m); B[0] = 1.0
    C = np.zeros(m); C[-1] = 1.0
    return A, B, C, np.ones(m)

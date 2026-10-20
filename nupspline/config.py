"""
Parameters of the proposed method (Sec. IV of the paper: fixed for all
scenarios, signals and K; chosen by hand, not tuned per signal).

Notation as in the paper:
  Model 1   X_{k+1} = A X_k + B U_k + B' U'_k,   Y~_k = C X_k + Z_k          (10), (6)
            U_k ~ N(0, q_k^2),  Z_k ~ N(0, r_k^2),  U'_k ~ p_beta1 (NUV, gamma_{1,k})
  Model 2   qbar_k = 1/q_k, piecewise constant: random walk with NUV(beta2)
            increments, NUV parameters gamma_{2,k}                          (Fig. 2)
  Model 3   rbar_k = 1/r_k, the same with beta3, gamma_{3,k}

With standardize=True (default) fit() centers the observations by their median and
divides them by the noise level sigma_hat estimated from the median absolute deviation
of their first differences (smoother.noise_level), fits, and transforms back.  The
initial values below are then dimensionless: in units of sigma_hat (qbar, rbar) and
sigma_hat^2 (gamma^2), so the method is invariant to the location and scale of the data.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    # ---- parameters stated in the paper (Sec. IV) -------------------------
    m: int = 2                      # Delta^m Y_k = U_k; m = 2: cubic smoothing splines
    beta1: float = 3.0              # NUV prior of the level jumps U'_k        (Model 1)
    beta2: float = 1.0              # NUV prior of the jumps of qbar_k         (Model 2)
    beta3: float = 2.0              # NUV prior of the jumps of rbar_k         (Model 3)
    alpha: float = float(np.sqrt(0.8))   # damping in (17)-(21): alpha^2 = 0.8
    standardize: bool = True        # center by the median, divide by sigma_hat, transform back
    qbar_init: float = 1.0          # initial qbar_k = 1/q_k for all k     (units of 1/sigma_hat)
    rbar_init: float = 1.0          # initial rbar_k = 1/r_k for all k     (units of 1/sigma_hat):
                                    #   q_k = r_k = sigma_hat, the estimated noise level
    gamma_sq_init: float = 1e3      # initial gamma_{1,k}^2 = gamma_{2,k}^2 = gamma_{3,k}^2
                                    #   (units of sigma_hat^2)
    constant_noise: bool = False    # True (Scenario 1): gamma_{3,k}^2 = 0 held fixed,
                                    # Model 3 is an equality chain -> one noise variance
    level_jumps: bool = True        # include the level-jump input B' U'_k
    n_iter: int = 1000              # fixed number of EM iterations

    # ---- implementation details (not stated in the paper) -----------------
    x0_var: float = 1e3             # prior variance of the initial state of Model 1 (times I)
    bar0_var: float = 1e6           # prior variance of the initial state of Models 2 and 3

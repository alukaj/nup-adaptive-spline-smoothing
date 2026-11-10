"""
Parameters of the proposed method (Secs. II-E and IV-B of the paper: fixed for all
scenarios, signals and K; chosen by hand, not tuned per signal).

Notation as in the paper (Fig. 1):
  X_{k+1} = A X_k + B U_k + B' U'_k,  Y~_k = C X_k + Z_k                  (9), (6)
  U_k ~ N(0, q_k^2),  Z_k ~ N(0, r_k^2),
  U'_k: level jump with NUP prior (10), variance sigma^2_{U'_k}, hyperparameter beta
  R^inv_k = 1/R_k:  R^inv_{k+1} = R^inv_k + S_k, NUP prior (10) on S_k with variance
                    sigma^2_{S_k} and hyperparameter beta_R                     (12)
  Q^inv_k = 1/Q_k:  the same with S'_k, sigma^2_{S'_k} and beta_Q
Estimates: r_hat_k = 1 / E[R^inv_k], q_hat_k = 1 / E[Q^inv_k] (Sec. III, Step 3b).

With standardize=True (default) fit() centers the observations by their median and
divides them by the noise level sigma_hat estimated from the median absolute deviation
of their first differences (smoother.noise_level), fits, and transforms back (Sec. IV-A).
The initial values below are then dimensionless: in units of 1/sigma_hat (R^inv, Q^inv)
and sigma_hat^2 (the variances sigma^2), so the method is invariant to the location and
scale of the data.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    # ---- parameters stated in the paper (Secs. II-E, III, IV-B) ------------
    m: int = 2                      # Delta^m Y_k = U_k (2); m = 2: cubic smoothing splines
    beta: float = 3.0               # NUP prior (10) of the level jumps U'_k
    beta_Q: float = 1.0             # NUP prior of the jumps S'_k of Q^inv_k
    beta_R: float = 2.0             # NUP prior of the jumps S_k of R^inv_k
    alpha: float = float(np.sqrt(0.8))   # in (13), (14): alpha^2 = 0.8
    standardize: bool = True        # center by the median, divide by sigma_hat, transform back
    qinv_init: float = 1.0          # initial Q^inv_k, i.e. q_hat_k = 1  (units of 1/sigma_hat)
    rinv_init: float = 1.0          # initial R^inv_k, i.e. r_hat_k = 1  (units of 1/sigma_hat):
                                    #   q_hat_k = r_hat_k = sigma_hat, the estimated noise level
    sigma2_init: float = 1e3        # initial sigma^2_{U'_k} = sigma^2_{S'_k} = sigma^2_{S_k}
                                    #   (units of sigma_hat^2)
    constant_noise: bool = False    # True (Scenario 1): S_k = 0 held fixed (sigma^2_{S_k} = 0),
                                    #   R^inv_k is one unknown constant
    level_jumps: bool = True        # include the level-jump input B' U'_k
    n_iter: int = 1000              # fixed number of EM iterations

    # ---- implementation details (not stated in the paper) -----------------
    x0_var: float = 1e3             # prior variance of the initial state X_0 (times I)
    inv0_var: float = 1e3           # prior variance of the initial R^inv_0 and Q^inv_0

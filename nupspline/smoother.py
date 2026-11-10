"""
The proposed method (Sec. III of the paper): MAP estimation of R^inv_k, Q^inv_k and
sigma^2_{U'_k} by EM (with U_k, U'_k, Z_k integrated out), and of sigma^2_{S_k},
sigma^2_{S'_k} in a second layer by EM as well, with Gaussian message passing via the
modified Bryson-Frazier (MBF) smoother; the signal estimate yhat_k = E[Y_k] is a
byproduct.  Notation as in the paper (Fig. 1), see config.py.

One EM iteration (the steps of Sec. III):
  Step 1   MBF in the state-space model (9), (6) with the current q_hat_k^2, r_hat_k^2
           and sigma^2_{U'_k}  ->  yhat_k = E[Y_k] and E[U_k^2], E[(U'_k)^2], E[Z_k^2].
  Step 2   sigma^2_{U'_k} = E[(U'_k)^2] / beta                                  (11)
  Step 3   a) backward messages of R^inv_k from E[Z_k^2]                   (13), (14)
           b) MBF in the model (12) of R^inv_k -> r_hat_k = 1 / E[R^inv_k] and E[S_k^2]
           c) sigma^2_{S_k} = E[S_k^2] / beta_R                                 (15)
  Step 4   the same for q_hat_k (Q^inv_k, S'_k, beta_Q) from E[U_k^2].

Indexing: inside the compiled functions every per-sample array has length K + 1
and index k is the paper's k = 1, ..., K; index 0 is a dummy (y[0], q2[0],
s2Up[0], mU[0], ... are not used, q2[0] = s2Up[0] = 0).  The state-space model
starts from X_0 ~ N(0, x0_var I), and step k predicts X_k = A X_{k-1} +
B U_{k-1} + B' U'_{k-1} and observes y~_k.  For k = 1 the input is the dummy
U_0 with variance q2[0] = 0, so X_1 = A X_0 and no input precedes the first
observation; U_k, U'_k act between y~_k and y~_{k+1}, and U_k = Delta^m Y_k as
in (2), k = 1, ..., K - m.  The inputs U_k with k > K - m would only affect
Y_{K+1}, ..., so they are set to zero (q2[k] = 0), and only Q^inv_1, ...,
Q^inv_{K-m} receive a backward message; the piecewise-constant model (12)
carries Q^inv_{K-m} forward to k > K - m.  The level jumps U'_k are likewise
restricted to k = 1, ..., K - m, as in Sec. II-C (sigma^2_{U'_k} = 0 for
k > K - m): a jump U'_k with k > K - m would only shift the last K - k < m
observations, which the inputs U_{k+1-m}, ..., U_{K-m} can reproduce exactly
(for m = 2, U'_{K-1} and U_{K-2} both move only y~_K), so it could not be told
apart from a change of the smooth part.
In the models (12) of R^inv_k and Q^inv_k, index k of the jump arrays holds the
jump into R^inv_k, i.e. S_{k-1} = R^inv_k - R^inv_{k-1} and sigma^2_{S_{k-1}};
the chain starts from R^inv_0 ~ N(0, inv0_var) (the same for Q^inv_k, S'_k).
fit() returns 0-based arrays of length K (index i holds k = i + 1).  With
sigma^2_{S_k} = 0 (Scenario 1) the EM update keeps it at 0, so R^inv_k is one
unknown constant and one noise variance r^2 is estimated.

Compiled with numba; the first call in a process compiles (a few seconds,
then cached on disk).
"""
import numpy as np
from numba import njit

from .config import Config
from .ssm import build_ssm


# ---------------------------------------------------------------------------
# Step 1: MBF smoother in the state-space model (9), (6)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _mbf_signal(y, A, B, Bp, r2, q2, s2Up, x0_var,
                yhat, vyhat, mU, VU, mUp, VUp,
                mXp, VXp, G, Ec):
    """MBF in the state-space model (9), (6) for given variances r2 = r_k^2,
    q2 = q_k^2, s2Up = sigma^2_{U'_k}.  Outputs: posterior mean / variance of Y_k (yhat, vyhat),
    of U_k (mU, VU) and of U'_k (mUp, VUp).  C = [0, ..., 0, 1].
    All arrays have length K + 1, index k = 1, ..., K (index 0: dummy,
    q2[0] = s2Up[0] = 0 for U_0 = U'_0 = 0)."""
    K = y.shape[0] - 1
    m = A.shape[0]
    c = m - 1                                   # Y_k = C X_k = last state component
    mX = np.zeros(m)                            # X_0 ~ N(0, x0_var I)
    VX = np.zeros((m, m))
    for p in range(m):
        VX[p, p] = x0_var
    tmp = np.empty((m, m))

    # forward (Kalman filter): prediction X_k = A X_{k-1} + B U_{k-1} + B' U'_{k-1}
    # (mXp, VXp) and gain G = 1 / (r_k^2 + C VXp C^T)
    for k in range(1, K + 1):
        for p in range(m):
            s = 0.0
            for l in range(m):
                s += A[p, l] * mX[l]
            mXp[k, p] = s
        for p in range(m):
            for j in range(m):
                s = 0.0
                for l in range(m):
                    s += A[p, l] * VX[l, j]
                tmp[p, j] = s
        for p in range(m):
            for j in range(m):
                s = 0.0
                for l in range(m):
                    s += tmp[p, l] * A[j, l]
                VXp[k, p, j] = s + B[p] * q2[k - 1] * B[j] + Bp[p] * s2Up[k - 1] * Bp[j]
        g = 1.0 / (r2[k] + VXp[k, c, c])
        G[k] = g
        e = g * (y[k] - mXp[k, c])
        Ec[k] = e
        for p in range(m):
            mX[p] = mXp[k, p] + VXp[k, p, c] * e
        for p in range(m):
            for j in range(m):
                VX[p, j] = VXp[k, p, j] - VXp[k, p, c] * g * VXp[k, c, j]

    # backward (MBF dual messages at the prediction, F = I - G VXp C^T C):
    #   xi'_k = F^T xi_k - C^T G (y_k - C mXp_k),   W'_k = F^T W_k F + C^T G C,
    #   xi_{k-1} = A^T xi'_k,                       W_{k-1} = A^T W'_k A
    xitX = np.zeros(m)
    WtX = np.zeros((m, m))
    xitXp = np.empty(m)
    WtXp = np.empty((m, m))
    M = np.empty((m, m))
    vM = np.empty(m)
    # U_K, U'_K act after the last observation: posterior = prior
    mU[K] = 0.0
    VU[K] = q2[K]
    mUp[K] = 0.0
    VUp[K] = s2Up[K]
    for k in range(K, 0, -1):
        g = G[k]
        vx = 0.0
        for j in range(m):
            vx += VXp[k, j, c] * xitX[j]
        for p in range(m):
            xitXp[p] = xitX[p]
        xitXp[c] -= g * vx
        xitXp[c] -= Ec[k]
        for p in range(m):
            s = 0.0
            for j in range(m):
                s += WtX[p, j] * VXp[k, j, c]
            for j in range(m):
                M[p, j] = WtX[p, j]
            M[p, c] -= g * s
        for j in range(m):
            s = 0.0
            for p in range(m):
                s += VXp[k, p, c] * M[p, j]
            vM[j] = s
        for p in range(m):
            for j in range(m):
                WtXp[p, j] = M[p, j]
        for j in range(m):
            WtXp[c, j] -= g * vM[j]
        WtXp[c, c] += g

        # posteriors: E[X] = mXp - VXp xi',  V[X] = VXp - VXp W' VXp,
        #             E[U] = -q^2 B^T xi',   V[U] = q^2 - q^2 B^T W' B q^2   (same for U' with B'),
        # for the input U_{k-1} that enters X_k (k = 1: the dummy U_0, which stays 0)
        s = 0.0
        for j in range(m):
            s += VXp[k, c, j] * xitXp[j]
        yhat[k] = mXp[k, c] - s
        s = 0.0
        for j in range(m):
            t = 0.0
            for l in range(m):
                t += WtXp[j, l] * VXp[k, l, c]
            s += VXp[k, c, j] * t
        vyhat[k] = VXp[k, c, c] - s
        bx = 0.0
        bWb = 0.0
        ax = 0.0
        aWa = 0.0
        for j in range(m):
            bx += xitXp[j] * B[j]
            ax += xitXp[j] * Bp[j]
            for l in range(m):
                bWb += B[j] * WtXp[j, l] * B[l]
                aWa += Bp[j] * WtXp[j, l] * Bp[l]
        mU[k - 1] = -q2[k - 1] * bx
        VU[k - 1] = q2[k - 1] - q2[k - 1] * bWb * q2[k - 1]
        mUp[k - 1] = -s2Up[k - 1] * ax
        VUp[k - 1] = s2Up[k - 1] - s2Up[k - 1] * aWa * s2Up[k - 1]

        for p in range(m):
            s = 0.0
            for l in range(m):
                s += A[l, p] * xitXp[l]
            xitX[p] = s
        for p in range(m):
            for j in range(m):
                s = 0.0
                for l in range(m):
                    s += WtXp[p, l] * A[l, j]
                tmp[p, j] = s
        for p in range(m):
            for j in range(m):
                s = 0.0
                for l in range(m):
                    s += A[l, p] * tmp[l, j]
                WtX[p, j] = s


# ---------------------------------------------------------------------------
# Steps 3 and 4: backward messages (13), (14), MBF in the model (12), update (15)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _update_inv(EV, VV, inv, xi_msg, s2, alpha, beta, inv0_var, k_last,
                W_msg, mXp, VXp, G, E, mS, VS):
    """Step 3 (inv = R^inv_k, EV/VV = posterior mean/variance of Z_k, s2 =
    sigma^2_{S_{k-1}}, beta = beta_R) or Step 4 (inv = Q^inv_k, of U_k, s2 =
    sigma^2_{S'_{k-1}}, beta = beta_Q).  inv, xi_msg and s2 are updated in place.
    All arrays have length K + 1, index k = 1, ..., K (index 0: dummy).  Only
    k = 1, ..., k_last send a backward message: K - m for Q^inv_k (U_k with
    k > K - m are not informed by the data), K for R^inv_k; for k > k_last the
    chain just continues from inv_{k_last}."""
    K = EV.shape[0] - 1
    a2 = alpha ** 2
    reg = alpha ** (-2) - 1.0
    # Step 3a: backward messages in information form, with r_hat_k = 1 / R^inv_k
    # (Step 4: E[U_k^2] and q_hat_k = 1 / Q^inv_k instead):
    #   W_k  = 1 / sigma_k^2 = E[Z_k^2] + (alpha^-2 - 1) r_hat_k^2               (13)
    #   xi_k = m_k / sigma_k^2 = alpha^-2 r_hat_k                                (14)
    for k in range(1, k_last + 1):
        if inv[k] != 0.0:
            xi_msg[k] = 1.0 / (a2 * inv[k])
        W_msg[k] = EV[k] * EV[k] + VV[k] + reg * inv[k] ** (-2)
        if W_msg[k] < 0.0:
            raise ValueError("negative backward-message precision")

    # Step 3b, MBF forward: R^inv_k = R^inv_{k-1} + S_{k-1} (12), S_{k-1} ~
    # N(0, sigma^2_{S_{k-1}}), R^inv_0 ~ N(0, inv0_var), observed through the
    # backward messages N(m_k, sigma_k^2)
    mX = 0.0
    VX = inv0_var
    for k in range(1, K + 1):
        mxp = mX
        vxp = VX + s2[k]
        if k <= k_last:
            sk2 = 1.0 / W_msg[k]
            mk = xi_msg[k] * sk2
            g = 1.0 / (sk2 + vxp)
            e = g * (mk - mxp)
        else:                                    # no message: prediction only
            g = 0.0
            e = 0.0
        mXp[k] = mxp
        VXp[k] = vxp
        G[k] = g
        E[k] = e
        mX = mxp + vxp * e
        VX = vxp - vxp * g * vxp

    # Step 3b, MBF backward: new E[R^inv_k] and the posterior of the jumps S_{k-1}
    xitX = 0.0
    WtX = 0.0
    for k in range(K, 0, -1):
        f = 1.0 - VXp[k] * G[k]
        xitXp = f * xitX - E[k]
        WtXp = f * WtX * f + G[k]
        mb = mXp[k] - VXp[k] * xitXp
        mS[k] = -s2[k] * xitXp
        v = s2[k] - s2[k] * WtXp * s2[k]
        VS[k] = v if v > 0.0 else 0.0
        if mb < 0.0:
            raise ValueError("negative estimate of R^inv_k or Q^inv_k")
        inv[k] = mb
        xitX = xitXp
        WtX = WtXp

    # Step 3c: sigma^2_{S_k} = E[S_k^2] / beta_R  (15)   (Step 4: S'_k, beta_Q; 0 stays 0)
    for k in range(1, K + 1):
        s2[k] = (VS[k] + abs(mS[k]) ** 2) / beta


# ---------------------------------------------------------------------------
# EM
# ---------------------------------------------------------------------------
@njit(cache=True)
def _em(y_obs, A, B, Bp, use_jumps, n_iter, qinv_init, rinv_init,
        s2Sp_init, s2S_init, s2Up_init, beta, beta_Q, beta_R,
        alpha, x0_var, inv0_var):
    """EM on the observations y_obs = y~_1..y~_K (0-based, length K); returns
    yhat_k, r_hat_k^2, q_hat_k^2, sigma^2_{U'_k} as 0-based arrays of length K.
    Inside, all per-sample arrays have length K + 1 and index k = 1, ..., K."""
    K = y_obs.shape[0]
    m = A.shape[0]
    y = np.zeros(K + 1)                          # y[k] = y~_k, y[0] dummy
    y[1:] = y_obs
    qinv = np.full(K + 1, qinv_init)             # E[Q^inv_k], q_hat_k = 1 / qinv[k]
    rinv = np.full(K + 1, rinv_init)             # E[R^inv_k], r_hat_k = 1 / rinv[k]
    xi_q = np.zeros(K + 1)
    xi_r = np.zeros(K + 1)
    s2Sp = np.full(K + 1, s2Sp_init)             # sigma^2_{S'_{k-1}} (jump into Q^inv_k)
    s2S = np.full(K + 1, s2S_init)               # sigma^2_{S_{k-1}}  (jump into R^inv_k)
    s2Up = np.zeros(K + 1)                       # sigma^2_{U'_k}, s2Up[0] = 0 (U'_0),
    if use_jumps:                                # s2Up[k] = 0 for k > K - m (U'_k = 0)
        s2Up[1:K - m + 1] = s2Up_init
    q = np.full(K + 1, 1.0 / qinv_init)          # q_hat_k
    r = np.full(K + 1, 1.0 / rinv_init)          # r_hat_k

    yhat = np.zeros(K + 1); vyhat = np.zeros(K + 1)
    mU = np.zeros(K + 1); VU = np.zeros(K + 1); mUp = np.zeros(K + 1); VUp = np.zeros(K + 1)
    mXp = np.zeros((K + 1, m)); VXp = np.zeros((K + 1, m, m)); G = np.zeros(K + 1); Ec = np.zeros(K + 1)
    r2 = np.zeros(K + 1); q2 = np.zeros(K + 1)   # q2[0] = 0 (U_0), q2[k] = 0 for k > K - m
    mZ = np.zeros(K + 1); VZ = np.zeros(K + 1)
    W_msg = np.zeros(K + 1); m2 = np.zeros(K + 1); v2 = np.zeros(K + 1); g2 = np.zeros(K + 1)
    e2 = np.zeros(K + 1); mS = np.zeros(K + 1); VS = np.zeros(K + 1)

    for _ in range(n_iter):
        for k in range(1, K - m + 1):            # U_k = 0 for k > K - m
            q2[k] = q[k] * 1.0 * q[k]
        for k in range(1, K + 1):
            r2[k] = r[k] * 1.0 * r[k]
        # Step 1: MBF in the state-space model (9), (6)
        _mbf_signal(y, A, B, Bp, r2, q2, s2Up, x0_var,
                    yhat, vyhat, mU, VU, mUp, VUp, mXp, VXp, G, Ec)
        # Step 2: sigma^2_{U'_k} = E[(U'_k)^2] / beta  (11)
        if use_jumps:
            for k in range(1, K - m + 1):        # U'_k = 0 for k > K - m
                s2Up[k] = (VUp[k] + abs(mUp[k]) ** 2) / beta
        # posterior of the noise Z_k = Y~_k - Y_k
        for k in range(1, K + 1):
            W = 1.0 / r2[k]
            WtYt = W - W * vyhat[k] * W
            xiYt = W * yhat[k] - W * y[k]
            mZ[k] = -(r2[k] * xiYt)
            VZ[k] = r2[k] - r2[k] * WtYt * r2[k]
        # Step 3: r_hat_k (R^inv_k from E[Z_k^2]);  Step 4: q_hat_k (Q^inv_k from E[U_k^2])
        _update_inv(mZ, VZ, rinv, xi_r, s2S, alpha, beta_R, inv0_var, K,
                    W_msg, m2, v2, g2, e2, mS, VS)
        for k in range(1, K + 1):
            r[k] = 1.0 / rinv[k]
        _update_inv(mU, VU, qinv, xi_q, s2Sp, alpha, beta_Q, inv0_var, K - m,
                    W_msg, m2, v2, g2, e2, mS, VS)
        for k in range(1, K + 1):
            q[k] = 1.0 / qinv[k]

    return yhat[1:], r[1:] * r[1:], q[1:] * q[1:], s2Up[1:]


def noise_level(y):
    """Robust estimate of the noise standard deviation from the first differences:

        sigma_hat = 1.4826 * MAD(diff(y)) / sqrt(2),   MAD(x) = median|x - median(x)|.

    1.4826 = 1 / Phi^{-1}(3/4) turns the MAD of Gaussian data into its standard
    deviation; sqrt(2) undoes the doubling of the noise variance by differencing.  The
    median ignores the few large differences at level jumps and steep peaks; for rapidly
    varying signals the estimate is biased upwards, which is harmless for its use as the
    scale of the initial values."""
    d = np.diff(np.asarray(y, float))
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2.0))


def fit(y, cfg=None):
    """Run the proposed method on the observations y~_1..y~_K.

    With cfg.standardize (default), the method is applied to (y - median(y)) / sigma_hat,
    sigma_hat = noise_level(y), and the estimates are transformed back, so that the
    initial values in cfg are dimensionless and the result is invariant to the location
    and scale of y.

    Returns dict (in the units of y):
      y_hat          signal estimate yhat_k                               (K,)
      r2             estimated noise variance r_hat_k^2                   (K,)
      q2             estimated smoothing parameter q_hat_k^2              (K,)
      sigma2_Uprime  estimated variance sigma^2_{U'_k} of the level jumps (K,)
      sigma_hat      the noise-level estimate used for standardization (1.0 without)
    Level jumps: detect_jumps(result), i.e. where sigma2_Uprime > q2 (Sec. IV-B).
    """
    cfg = cfg or Config()
    y = np.ascontiguousarray(np.asarray(y, float).ravel())
    loc, scale = 0.0, 1.0
    if cfg.standardize:
        loc, scale = float(np.median(y)), noise_level(y)
        if not (np.isfinite(scale) and scale > 0):       # e.g. constant or tiny series
            scale = float(np.std(y)) if np.std(y) > 0 else 1.0
        y = (y - loc) / scale
    A, B, C, Bp = build_ssm(cfg.m)
    if not cfg.level_jumps:
        Bp = np.zeros_like(Bp)
    y_hat, r2, q2, s2Up = _em(
        y, A, B, Bp, bool(cfg.level_jumps), int(cfg.n_iter),
        float(cfg.qinv_init), float(cfg.rinv_init),
        float(cfg.sigma2_init),                                      # sigma^2_{S'_k}
        0.0 if cfg.constant_noise else float(cfg.sigma2_init),       # sigma^2_{S_k}
        float(cfg.sigma2_init),                                      # sigma^2_{U'_k}
        float(cfg.beta), float(cfg.beta_Q), float(cfg.beta_R), float(cfg.alpha),
        float(cfg.x0_var), float(cfg.inv0_var))
    s2 = scale * scale
    return {"y_hat": loc + scale * y_hat, "r2": s2 * r2, "q2": s2 * q2,
            "sigma2_Uprime": s2 * s2Up, "sigma_hat": scale}


def jump_score(est):
    """Per-sample jump score s, s[k] for a jump between y[k-1] and y[k] (0-based, as
    jumps.true_jumps): U'_k (array index k - 1) lies between y[k-1] and y[k], so
    s[k] = sigma^2_{U'_k} / q_hat_k^2 for k >= 1 and s[0] = 0 (no jump before y[0]);
    U'_K, after the last observation, is dropped.  Since sigma^2_{U'_k} = 0 for
    k > K - m, s[k] = 0 for the last m - 1 samples."""
    g = np.asarray(est["sigma2_Uprime"]) / np.asarray(est["q2"])
    return np.concatenate(([0.0], g[:-1]))


def detect_jumps(est):
    """Indices k of the detected level jumps (between y[k-1] and y[k], 0-based):
    sigma^2_{U'_k} > q_hat_k^2 (Sec. IV-B), which is scale-free and has no tuning
    parameter."""
    return np.flatnonzero(jump_score(est) > 1.0)


def warm_up():
    """Compile (or load from cache) the numba kernels, so that timed fits exclude it."""
    fit(np.sin(np.linspace(0.0, 3.0, 32)), Config(n_iter=2))

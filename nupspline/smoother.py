"""
The proposed method (Sec. III of the paper): type-II MAP estimation of
theta = (qbar, rbar, gamma_1, gamma_2, gamma_3) by EM, with Gaussian message
passing via the modified Bryson-Frazier (MBF) smoother in all three models,
followed by the signal estimate yhat_k.

One EM iteration:
  E-step   MBF in Model 1 (eqs. (10), (6)) with the current q_k^2 = qbar_k^-2,
           r_k^2 = rbar_k^-2 and gamma_{1,k}^2  ->  posterior moments of
           U_k, U'_k, Z_k and the signal Y_k.
  M-step   gamma_{1,k}^2 = E[U'_k^2] / beta1                               (9)
           Model 2: backward messages of qbar_k from E[U_k^2]           (18), (19)
                    MBF in Model 2 -> new qbar_k (16) and the posterior of
                    its increments ~U_k; gamma_{2,k}^2 = E[~U_k^2] / beta2  (9)
           Model 3: the same for rbar_k from E[Z_k^2]                   (20)-(22)
Models 2 and 3 (Fig. 2): Qbar'_k = Qbar'_{k-1} + ~U_k, ~U_k ~ N(0, gamma_{2,k}^2),
Qbar_k = Qbar'_k (and the same for rbar with gamma_3).

Implementation notes: a prior X_0 ~ N(0, x0_var I) precedes the first
observation in Model 1; Models 2 and 3 start from N(0, bar0_var); the NUV
parameters gamma_{2,k}^2, gamma_{3,k}^2 are capped at gamma_sq_cap.  With
gamma_{3,k}^2 = 0 (Scenario 1) the EM update keeps it at 0, so Model 3 is an
equality chain and one noise variance r^2 is estimated.

Compiled with numba; the first call in a process compiles (a few seconds,
then cached on disk).
"""
import numpy as np
from numba import njit

from .config import Config
from .ssm import build_ssm


# ---------------------------------------------------------------------------
# Model 1: MBF smoother
# ---------------------------------------------------------------------------
@njit(cache=True)
def _mbf_model1(y, A, B, Bp, r2, q2, g1sq, x0_var,
                yhat, vyhat, mU, VU, mUp, VUp,
                mXp, VXp, G, Ec):
    """MBF in Model 1 for given variances r2 = r_k^2, q2 = q_k^2, g1sq =
    gamma_{1,k}^2.  Outputs: posterior mean / variance of Y_k (yhat, vyhat),
    of U_k (mU, VU) and of U'_k (mUp, VUp).  C = [0, ..., 0, 1]."""
    K = y.shape[0]
    m = A.shape[0]
    c = m - 1                                   # Y_k = C X_k = last state component
    mX = np.zeros(m)
    VX = np.zeros((m, m))
    for p in range(m):
        VX[p, p] = x0_var
    tmp = np.empty((m, m))

    # forward (Kalman filter): prediction mXp, VXp and gain G = 1 / (r_k^2 + C VXp C^T)
    for i in range(K):
        for p in range(m):
            s = 0.0
            for l in range(m):
                s += A[p, l] * mX[l]
            mXp[i, p] = s
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
                VXp[i, p, j] = s + B[p] * q2[i] * B[j] + Bp[p] * g1sq[i] * Bp[j]
        g = 1.0 / (r2[i] + VXp[i, c, c])
        G[i] = g
        e = g * (y[i] - mXp[i, c])
        Ec[i] = e
        for p in range(m):
            mX[p] = mXp[i, p] + VXp[i, p, c] * e
        for p in range(m):
            for j in range(m):
                VX[p, j] = VXp[i, p, j] - VXp[i, p, c] * g * VXp[i, c, j]

    # backward (MBF dual messages at the prediction, F = I - G VXp C^T C):
    #   xi'_k = F^T xi_k - C^T G (y_k - C mXp_k),   W'_k = F^T W_k F + C^T G C,
    #   xi_{k-1} = A^T xi'_k,                       W_{k-1} = A^T W'_k A
    xitX = np.zeros(m)
    WtX = np.zeros((m, m))
    xitXp = np.empty(m)
    WtXp = np.empty((m, m))
    M = np.empty((m, m))
    vM = np.empty(m)
    for i in range(K - 1, -1, -1):
        g = G[i]
        vx = 0.0
        for j in range(m):
            vx += VXp[i, j, c] * xitX[j]
        for p in range(m):
            xitXp[p] = xitX[p]
        xitXp[c] -= g * vx
        xitXp[c] -= Ec[i]
        for p in range(m):
            s = 0.0
            for j in range(m):
                s += WtX[p, j] * VXp[i, j, c]
            for j in range(m):
                M[p, j] = WtX[p, j]
            M[p, c] -= g * s
        for j in range(m):
            s = 0.0
            for p in range(m):
                s += VXp[i, p, c] * M[p, j]
            vM[j] = s
        for p in range(m):
            for j in range(m):
                WtXp[p, j] = M[p, j]
        for j in range(m):
            WtXp[c, j] -= g * vM[j]
        WtXp[c, c] += g

        # posteriors: E[X] = mXp - VXp xi',  V[X] = VXp - VXp W' VXp,
        #             E[U] = -q^2 B^T xi',   V[U] = q^2 - q^2 B^T W' B q^2   (same for U' with B')
        s = 0.0
        for j in range(m):
            s += VXp[i, c, j] * xitXp[j]
        yhat[i] = mXp[i, c] - s
        s = 0.0
        for j in range(m):
            t = 0.0
            for l in range(m):
                t += WtXp[j, l] * VXp[i, l, c]
            s += VXp[i, c, j] * t
        vyhat[i] = VXp[i, c, c] - s
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
        mU[i] = -q2[i] * bx
        VU[i] = q2[i] - q2[i] * bWb * q2[i]
        mUp[i] = -g1sq[i] * ax
        VUp[i] = g1sq[i] - g1sq[i] * aWa * g1sq[i]

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
# Models 2 and 3: backward messages (18)-(22), MBF, NUV update (9)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _update_model23(EV, VV, bar, xi_msg, gsq, alpha, beta, bar0_var, cap,
                    W_msg, mXp, VXp, G, E, mUt, VUt):
    """One M-step of Model 2 (bar = qbar_k, EV/VV = posterior mean/variance of
    U_k) or Model 3 (bar = rbar_k, of Z_k).  bar, xi_msg and gsq (gamma_k^2)
    are updated in place."""
    K = EV.shape[0]
    a2 = alpha ** 2
    reg = alpha ** (-2) - 1.0
    # backward messages in information form:
    #   W_k  = 1 / sigma_k^2 = E[U_k^2] + (alpha^-2 - 1) bar_k^-2            (18)/(21)
    #   xi_k = m_k / sigma_k^2 = alpha^-2 bar_k^-1                            (19)/(22)
    for k in range(K):
        if bar[k] != 0.0:
            xi_msg[k] = 1.0 / (a2 * bar[k])
        W_msg[k] = EV[k] * EV[k] + VV[k] + reg * bar[k] ** (-2)
        if W_msg[k] < 0.0:
            raise ValueError("negative backward-message precision")

    # MBF forward: Bar'_k = Bar'_{k-1} + ~U_k, ~U_k ~ N(0, gamma_k^2), observed
    # through the backward messages N(m_k, sigma_k^2)
    mX = 0.0
    VX = bar0_var
    for i in range(K):
        mxp = mX
        vxp = VX + gsq[i]
        s2 = 1.0 / W_msg[i]
        mk = xi_msg[i] * s2
        g = 1.0 / (s2 + vxp)
        e = g * (mk - mxp)
        mXp[i] = mxp
        VXp[i] = vxp
        G[i] = g
        E[i] = e
        mX = mxp + vxp * e
        VX = vxp - vxp * g * vxp

    # MBF backward: new bar_k (16)/(20) and the posterior of the increments ~U_k
    xitX = 0.0
    WtX = 0.0
    for i in range(K - 1, -1, -1):
        f = 1.0 - VXp[i] * G[i]
        xitXp = f * xitX - E[i]
        WtXp = f * WtX * f + G[i]
        mb = mXp[i] - VXp[i] * xitXp
        mUt[i] = -gsq[i] * xitXp
        v = gsq[i] - gsq[i] * WtXp * gsq[i]
        VUt[i] = v if v > 0.0 else 0.0
        if mb < 0.0:
            raise ValueError("negative estimate of qbar_k or rbar_k")
        bar[i] = mb
        xitX = xitXp
        WtX = WtXp

    # NUV update (9): gamma_k^2 = E[~U_k^2] / beta   (0 stays 0)
    for i in range(K):
        v = (VUt[i] + abs(mUt[i]) ** 2) / beta
        gsq[i] = v if v < cap else cap


# ---------------------------------------------------------------------------
# EM
# ---------------------------------------------------------------------------
@njit(cache=True)
def _em(y, A, B, Bp, use_jumps, n_iter, qbar_init, rbar_init,
        g2_init, g3_init, g1_init, beta1, beta2, beta3,
        alpha, x0_var, bar0_var, cap):
    K = y.shape[0]
    m = A.shape[0]
    qbar = np.full(K, qbar_init)               # Model 2 variable, q_k = 1 / qbar_k
    rbar = np.full(K, rbar_init)               # Model 3 variable, r_k = 1 / rbar_k
    xi_q = np.zeros(K)
    xi_r = np.zeros(K)
    g2sq = np.full(K, g2_init)                 # gamma_{2,k}^2
    g3sq = np.full(K, g3_init)                 # gamma_{3,k}^2
    g1sq = np.full(K, g1_init) if use_jumps else np.zeros(K)   # gamma_{1,k}^2
    q = np.full(K, 1.0 / qbar_init)            # q_k
    r = np.full(K, 1.0 / rbar_init)            # r_k

    yhat = np.empty(K); vyhat = np.empty(K)
    mU = np.empty(K); VU = np.empty(K); mUp = np.zeros(K); VUp = np.zeros(K)
    mXp = np.empty((K, m)); VXp = np.empty((K, m, m)); G = np.empty(K); Ec = np.empty(K)
    r2 = np.empty(K); q2 = np.empty(K); mZ = np.empty(K); VZ = np.empty(K)
    W_msg = np.empty(K); m2 = np.empty(K); v2 = np.empty(K); g2 = np.empty(K); e2 = np.empty(K)
    mUt = np.empty(K); VUt = np.empty(K)

    for _ in range(n_iter):
        for k in range(K):
            q2[k] = q[k] * 1.0 * q[k]
            r2[k] = r[k] * 1.0 * r[k]
        # E-step: MBF in Model 1
        _mbf_model1(y, A, B, Bp, r2, q2, g1sq, x0_var,
                    yhat, vyhat, mU, VU, mUp, VUp, mXp, VXp, G, Ec)
        # M-step, Model 1: gamma_{1,k}^2 = E[U'_k^2] / beta1  (9)
        if use_jumps:
            for k in range(K):
                g1sq[k] = (VUp[k] + abs(mUp[k]) ** 2) / beta1
        # posterior of the noise Z_k = Y~_k - Y_k
        for k in range(K):
            W = 1.0 / r2[k]
            WtYt = W - W * vyhat[k] * W
            xiYt = W * yhat[k] - W * y[k]
            mZ[k] = -(r2[k] * xiYt)
            VZ[k] = r2[k] - r2[k] * WtYt * r2[k]
        # M-step, Model 3 (rbar from E[Z_k^2]) and Model 2 (qbar from E[U_k^2])
        _update_model23(mZ, VZ, rbar, xi_r, g3sq, alpha, beta3, bar0_var, cap,
                        W_msg, m2, v2, g2, e2, mUt, VUt)
        for k in range(K):
            r[k] = 1.0 / rbar[k]
        _update_model23(mU, VU, qbar, xi_q, g2sq, alpha, beta2, bar0_var, cap,
                        W_msg, m2, v2, g2, e2, mUt, VUt)
        for k in range(K):
            q[k] = 1.0 / qbar[k]

    return yhat, r * r, q * q, g1sq


def fit(y, cfg=None):
    """Run the proposed method on the observations y~_1..y~_K.

    Returns dict:
      y_hat      signal estimate yhat_k                                   (K,)
      r2         estimated noise variance r_k^2                          (K,)
      q2         estimated smoothing parameter q_k^2                     (K,)
      gamma1_sq  estimated gamma_{1,k}^2; level jumps where > 1         (K,)
    """
    cfg = cfg or Config()
    y = np.ascontiguousarray(np.asarray(y, float).ravel())
    A, B, C, Bp = build_ssm(cfg.m)
    if not cfg.level_jumps:
        Bp = np.zeros_like(Bp)
    y_hat, r2, q2, g1sq = _em(
        y, A, B, Bp, bool(cfg.level_jumps), int(cfg.n_iter),
        float(cfg.qbar_init), float(cfg.rbar_init),
        float(cfg.gamma_sq_init),                                      # gamma_{2,k}^2
        0.0 if cfg.constant_noise else float(cfg.gamma_sq_init),       # gamma_{3,k}^2
        float(cfg.gamma_sq_init),                                      # gamma_{1,k}^2
        float(cfg.beta1), float(cfg.beta2), float(cfg.beta3), float(cfg.alpha),
        float(cfg.x0_var), float(cfg.bar0_var), float(cfg.gamma_sq_cap))
    return {"y_hat": y_hat, "r2": r2, "q2": q2, "gamma1_sq": g1sq}


def warm_up():
    """Compile (or load from cache) the numba kernels, so that timed fits exclude it."""
    fit(np.sin(np.linspace(0.0, 3.0, 32)), Config(n_iter=2))

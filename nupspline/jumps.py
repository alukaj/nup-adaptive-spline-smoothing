"""
Scoring of the detected level jumps (Table II of the paper).

True jumps: Blocks (11) and HeaviSine (2, at t = 0.3 and 0.72); a jump at position p
lies between y_{k-1} and y_k for the first k with t_k > p.  Bumps and Doppler have no
jumps, so every detection on them is a false positive.

Detections from a per-sample score s_k and a threshold c:
  proposed method   s_k = sigma^2_{U'_k} / q_hat_k^2,  jump where s_k > 1 (Sec. IV-B)
  ABCO              s_k = posterior changepoint probability,  jump where s_k >= c
Every flagged sample is a detection; adjacent flagged samples are not merged.

Hits: a detection within TOL = 5 samples of a true jump, each detection and each true
jump used at most once, with the pairing that gives the most hits (maximum bipartite
matching), so of two detections near the same true jump one is a hit and the other a
false positive.  TP = hits, FP = detections - TP, FN = true jumps - TP.  The pooled F1 score
2 TP / (2 TP + FP + FN) sums TP, FP and FN over all signals and realizations.
"""
import numpy as np

JUMP_POSITIONS = {
    "blocks": (0.10, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81),
    "heavisine": (0.30, 0.72),
    "bumps": (),
    "doppler": (),
}
TOL = 5
ABCO_THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 20))   # 0.05, 0.10, ..., 0.95


def true_jumps(signal, K):
    """Indices k of the true jumps (between y_{k-1} and y_k) on t_k = (k-1)/(K-1)."""
    t = np.linspace(0.0, 1.0, int(K))
    return np.searchsorted(t, np.asarray(JUMP_POSITIONS[signal], float), side="right")


def detect(score, thr, strict=False):
    """Detected jumps: indices k with score > thr (strict) or >= thr; NaN never flags."""
    s = np.nan_to_num(np.asarray(score, float), nan=-np.inf)
    return np.flatnonzero(s > thr if strict else s >= thr)


def n_hits(det, true, tol=TOL):
    det, true = np.sort(det), np.sort(true)
    i = j = hits = 0
    while i < len(det) and j < len(true):
        if true[j] < det[i] - tol:      # true jump too far left of this and all later detections: missed
            j += 1
        elif true[j] > det[i] + tol:    # no remaining true jump near this detection: false positive
            i += 1
        else:                           # within tol: match the leftmost pair
            hits += 1; i += 1; j += 1
    return hits


def counts(det, true, tol=TOL):
    """(TP, FP, FN) of one series."""
    tp = n_hits(det, true, tol)
    return tp, len(det) - tp, len(true) - tp


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else float("nan")

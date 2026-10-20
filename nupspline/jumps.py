"""
Scoring of the detected level jumps (Table II of the paper).

True jumps: Blocks (11) and HeaviSine (2, at t = 0.3 and 0.72); a jump at position p
lies between y_{k-1} and y_k for the first k with t_k > p.  Bumps and Doppler have no
jumps, so every detection on them is a false positive.

Detections from a per-sample score s_k and a threshold c:
  proposed method   s_k = gamma_{1,k}^2 / q_k^2,  jump where s_k > 1
  ABCO              s_k = posterior changepoint probability,  jump where s_k >= c
Two adjacent detections count as one: taking the flagged samples from the largest
score down, each one not yet used becomes a detection and absorbs at most one flagged
neighbour.  (A single jump often flags two neighbours, e.g. ABCO with m = 2 always
does, as a +J / -J pair of second differences.)

Hits: a detection within TOL = 5 samples of a true jump, each detection and each true
jump used at most once, with the pairing that gives the most hits (maximum bipartite
matching).  TP = hits, FP = detections - TP, FN = true jumps - TP.  The pooled F1 score
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
ABCO_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def true_jumps(signal, K):
    """Indices k of the true jumps (between y_{k-1} and y_k) on t_k = (k-1)/(K-1)."""
    t = np.linspace(0.0, 1.0, int(K))
    return np.searchsorted(t, np.asarray(JUMP_POSITIONS[signal], float), side="right")


def detect(score, thr, strict=False, merge=True):
    """Detected jumps: score > thr (strict) or >= thr; NaN never flags.  merge: two
    adjacent flagged samples are one detection (strongest first, each detection absorbs
    at most one flagged neighbour)."""
    s = np.nan_to_num(np.asarray(score, float), nan=-np.inf)
    idx = np.flatnonzero(s > thr if strict else s >= thr)
    if not merge or idx.size < 2:
        return idx
    flagged, used, keep = set(idx.tolist()), set(), []
    for i in idx[np.argsort(-s[idx], kind="stable")].tolist():
        if i in used:
            continue
        keep.append(i)
        used.add(i)
        nb = [j for j in (i - 1, i + 1) if j in flagged and j not in used]
        if nb:
            used.add(max(nb, key=lambda j: s[j]))
    return np.sort(np.array(keep, dtype=int))


def n_hits(det, true, tol=TOL):
    """Largest number of one-to-one hits within tol samples (augmenting paths)."""
    det, true = np.asarray(det), np.asarray(true)
    if det.size == 0 or true.size == 0:
        return 0
    near = [np.flatnonzero(np.abs(true - d) <= tol).tolist() for d in det]
    owner = {}

    def augment(i, seen):
        for j in near[i]:
            if j not in seen:
                seen.add(j)
                if j not in owner or augment(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    return sum(augment(i, set()) for i in range(det.size))


def counts(det, true, tol=TOL):
    """(TP, FP, FN) of one series."""
    tp = n_hits(det, true, tol)
    return tp, len(det) - tp, len(true) - tp


def f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return 2 * tp / d if d else float("nan")

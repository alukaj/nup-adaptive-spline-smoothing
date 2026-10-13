"""
Test signals and data of the simulation study (Sec. IV).

The Donoho-Johnstone signals [1] (Blocks, Bumps, HeaviSine, Doppler) on
t_k = (k-1)/(K-1), k = 1..K, scaled to standard deviation 7, observed with
Gaussian noise of variance
  Scenario 1:  1 for all t
  Scenario 2:  1 on [0, 0.33), 16 on [0.33, 0.66), 1 on [0.66, 1].

Realization i of (scenario, signal, K) is drawn with
    numpy.random.default_rng([20240615, scenario - 1, signal index, K, i])
(signal index: heavisine 0, bumps 1, blocks 2, doppler 3), so every method,
process and machine sees exactly the same data.
"""
import numpy as np

SIGNALS = ("blocks", "bumps", "heavisine", "doppler")          # order of Table I / Fig. 3
TITLES = {"blocks": "Blocks", "bumps": "Bumps", "heavisine": "HeaviSine", "doppler": "Doppler"}
NOISE_SD = {1: (1.0, 1.0, 1.0), 2: (1.0, 4.0, 1.0)}              # per segment, per scenario
SEGMENT_BOUNDS = (0.33, 0.66)
BASE_SEED = 20240615
_SEED_INDEX = {"heavisine": 0, "bumps": 1, "blocks": 2, "doppler": 3}


def blocks(t):
    pos = np.array([0.10, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81])
    hgt = np.array([4.0, -5.0, 3.0, -4.0, 5.0, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2])
    return np.sum([h * (0.5 + 0.5 * np.sign(t - p)) for p, h in zip(pos, hgt)], axis=0)


def bumps(t):
    pos = np.array([0.10, 0.13, 0.15, 0.23, 0.25, 0.40, 0.44, 0.65, 0.76, 0.78, 0.81])
    hgt = np.array([4.0, 5.0, 3.0, 4.0, 5.0, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2])
    wid = np.array([0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005])
    return np.sum([h / (1 + abs((t - p) / w)) ** 4 for p, h, w in zip(pos, hgt, wid)], axis=0)


def heavisine(t):
    return 4 * np.sin(4 * np.pi * t) - np.sign(t - 0.3) - np.sign(0.72 - t)


def doppler(t):
    eps = 0.05
    return np.sqrt(t * (1 - t)) * np.sin(2 * np.pi * (1 + eps) / (t + eps))


_FUNCTIONS = {"blocks": blocks, "bumps": bumps, "heavisine": heavisine, "doppler": doppler}


def ground_truth(signal, K, scenario):
    """(t, y_true, r_true): grid, true signal y*_k (SD 7), true noise SD r_k."""
    t = np.linspace(0.0, 1.0, K)
    f = _FUNCTIONS[signal](t)
    y_true = f * (7.0 / f.std())
    cut = [int(round(b * K)) for b in SEGMENT_BOUNDS]
    r_true = np.empty(K)
    for a, b, sd in zip([0] + cut, cut + [K], NOISE_SD[scenario]):
        r_true[a:b] = sd
    return t, y_true, r_true


def realization(signal, K, scenario, i):
    """(t, y_obs, y_true, r_true) of realization i (y_obs = noisy observations y~_k)."""
    t, y_true, r_true = ground_truth(signal, K, scenario)
    rng = np.random.default_rng([BASE_SEED, scenario - 1, _SEED_INDEX[signal], K, i])
    return t, y_true + rng.normal(0.0, r_true), y_true, r_true

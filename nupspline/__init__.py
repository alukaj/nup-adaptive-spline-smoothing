"""Adaptive smoothing of time series with level jumps and time-varying noise
via NUP priors -- code for the simulations of the paper."""
from .config import Config
from .smoother import detect_jumps, fit, jump_score, noise_level, warm_up
from .signals import SIGNALS, TITLES, ground_truth, realization

__all__ = ["Config", "fit", "detect_jumps", "jump_score", "noise_level", "warm_up", "SIGNALS", "TITLES",
           "ground_truth", "realization"]

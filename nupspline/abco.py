"""
Competitor ABCO, adaptive Bayesian changepoints with outliers [10], from the
authors' R package dsp [22] (called through Rscript), without its outlier
component:

  m = 1 or 2 (dsp's D, the order of the differences of the trend)
  constant noise variance (Scenario 1) or ASV [9] (Scenario 2)

dsp's default settings (6000 iterations, 1000 burn-in).  A jump is declared at
k where the posterior changepoint probability is >= c (dsp's default c = 0.5).
The data are standardised before the fit and the estimate transformed back; the
seed is derived from the data, as for the BTF methods (btf.py).
"""
import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np

from .btf import rscript

R_SCRIPT = Path(__file__).resolve().parent / "dsp_abco.R"


def fit_abco(y, m=1, obs_sv="const", nsave=1000, nburn=1000, nskip=4, seed=None):
    """Fit ABCO with dsp.

    Returns dict: y_hat (posterior mean of the trend), cp_prob (posterior
    changepoint probability per k, NaN for the first m entries), q2 (posterior
    mean of the evolution variance of the m-th differences, NaN for the first m
    entries), r2 (posterior mean of the noise variance per k), seconds (run time
    of dsp_fit inside R)."""
    if m not in (1, 2):
        raise ValueError("m must be 1 or 2")
    y = np.asarray(y, float).ravel()
    if seed is None:
        seed = zlib.crc32(np.ascontiguousarray(y).tobytes()) % (2 ** 31 - 1)
    loc, scale = float(np.mean(y)), float(np.std(y, ddof=1))
    with tempfile.TemporaryDirectory(prefix="dsp_") as tmp:
        f_in, f_out = Path(tmp) / "y.txt", Path(tmp) / "out.txt"
        np.savetxt(f_in, (y - loc) / scale, fmt="%.17g")
        r = subprocess.run([rscript(), str(R_SCRIPT), str(f_in), str(f_out), str(m), obs_sv,
                            str(nsave), str(nburn), str(nskip), str(int(seed))],
                           capture_output=True, text=True)
        if r.returncode != 0 or not f_out.exists():
            raise RuntimeError(f"dsp failed (ABCO, m={m}, obsSV={obs_sv}):\n{r.stderr[-2000:]}")
        out = np.genfromtxt(f_out, missing_values="NA", filling_values=np.nan, ndmin=2)
    seconds = next((float(l.split()[1]) for l in r.stdout.splitlines()
                    if l.startswith("ELAPSED ")), float("nan"))
    return {"y_hat": loc + scale * out[:, 0], "cp_prob": out[:, 1],
            "q2": scale ** 2 * out[:, 2], "r2": scale ** 2 * out[:, 3], "seconds": seconds}

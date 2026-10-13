"""
Competitors: Bayesian trend filtering with dynamic horseshoe prior, from the
authors' R package dsp [21] (called through Rscript):

  BTF-DHS      [8]  constant noise variance               (Scenario 1)
  BTF-DHS-SV   [8]  log r_k^2 varies smoothly (SV)        (Scenario 2)
  BTF-DHS-ASV  [9]  horseshoe on the differences of log r_k^2 (ASV)  (Scenario 2)

All with dsp's default settings (nsave = 1000, nburn = 1000, nskip = 4:
6000 iterations, 1000 burn-in), D = 2.  The data are standardised before the
fit and the estimates transformed back; the seed is derived from the data, so
every fit is reproducible.  Requires R (>= 4.1) with the dsp package; Rscript
is taken from $RSCRIPT or the PATH.
"""
import os
import shutil
import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np

R_SCRIPT = Path(__file__).resolve().parent / "dsp_btf.R"
VARIANTS = {"btf_dhs": "const", "btf_dhs_sv": "SV", "btf_dhs_asv": "ASV"}


def rscript():
    exe = os.environ.get("RSCRIPT") or shutil.which("Rscript")
    if not exe:
        raise RuntimeError("Rscript not found: install R and the dsp package "
                           "(Rscript install_r_packages.R), or set $RSCRIPT")
    return exe


def check_dsp():
    """Return the installed dsp version, or raise with an explanation."""
    r = subprocess.run([rscript(), "-e", 'suppressPackageStartupMessages(library(dsp)); '
                        'cat(as.character(packageVersion("dsp")))'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("R package dsp not available -- run: Rscript install_r_packages.R\n"
                           + r.stderr)
    return r.stdout.strip()


def fit_btf(y, obs_sv="const", nsave=1000, nburn=1000, nskip=4, seed=None):
    """Fit BTF-DHS with dsp.  obs_sv: "const", "SV" or "ASV".

    Returns dict: y_hat (posterior mean of the signal), r2 (posterior mean of
    the noise variance per k), seconds (run time of dsp_fit inside R)."""
    y = np.asarray(y, float).ravel()
    if seed is None:
        seed = zlib.crc32(np.ascontiguousarray(y).tobytes()) % (2 ** 31 - 1)
    loc, scale = float(np.mean(y)), float(np.std(y, ddof=1))
    with tempfile.TemporaryDirectory(prefix="dsp_") as tmp:
        f_in, f_out = Path(tmp) / "y.txt", Path(tmp) / "out.txt"
        np.savetxt(f_in, (y - loc) / scale, fmt="%.17g")
        r = subprocess.run([rscript(), str(R_SCRIPT), str(f_in), str(f_out), obs_sv,
                            str(nsave), str(nburn), str(nskip), str(int(seed))],
                           capture_output=True, text=True)
        if r.returncode != 0 or not f_out.exists():
            raise RuntimeError(f"dsp failed (obsSV={obs_sv}):\n{r.stderr[-2000:]}")
        out = np.loadtxt(f_out, ndmin=2)
    seconds = next((float(l.split()[1]) for l in r.stdout.splitlines()
                    if l.startswith("ELAPSED ")), float("nan"))
    return {"y_hat": loc + scale * out[:, 0], "r2": scale ** 2 * out[:, 1],
            "q2": scale ** 2 * out[:, 2], "seconds": seconds}

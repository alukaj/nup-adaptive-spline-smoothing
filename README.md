# Adaptive Smoothing of Time Series with Level Jumps and Time-Varying Noise via NUP Priors

Code for the simulations of the paper

> A. Lukaj and H.-A. Loeliger, *Adaptive Smoothing of Time Series with Level Jumps and
> Time-Varying Noise via NUP Priors.*

It reproduces **Table I** (average MSE and run time), **Table II** (F1 score of the
detected level jumps) and **Fig. 3** (one realization of Scenario 2) of the paper.

## Contents

```
nupspline/                 the proposed method, the data, the competitors, the scoring
  config.py                parameters of the proposed method (Sec. IV)
  ssm.py                   state-space model (5)-(7), level-jump input B' (10)
  smoother.py              EM with MBF message passing in Models 1, 2, 3 (Sec. III);
                           standardization of the data; level-jump detection
  signals.py               Donoho-Johnstone signals, Scenarios 1 and 2, seeds
  btf.py, dsp_btf.R        competitors BTF-DHS / -SV / -ASV [8], [9] via the R package dsp [22]
  abco.py, dsp_abco.R      competitor ABCO [10] via dsp
  jumps.py                 scoring of the detected level jumps (Table II)
run_simulations.py         runs the study (all methods, scenarios, signals, K)
time_fits.py               run times of Table I (one fit at a time)
make_table.py              builds Tables I and II (LaTeX + console)
make_figure.py             builds Fig. 3
install_r_packages.R       installs the R package dsp
```

## Installation

**Python** (3.10 or newer):

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt     # numpy, numba, matplotlib
```

**R** (only for the competitors; R >= 4.1):

```bash
Rscript install_r_packages.R        # installs dsp from CRAN (the paper uses dsp 1.6.0)
```

Without R, the proposed method can still be run on its own (`--methods proposed`).

## Quick check (a few minutes)

```bash
python run_simulations.py --quick                # K = 128, 5 realizations, all methods
python make_table.py --results results_quick     # prints Tables I and II, writes tables.tex
python make_figure.py --out results_quick/figure3.pdf
```

Without R: `python run_simulations.py --quick --methods proposed`.

## Full reproduction of the paper

```bash
python run_simulations.py          # 2 scenarios x 4 signals x K in {128, 512, 1024} x 100 realizations
python time_fits.py                # run times of Table I: 40 fits per method and K, one at a time
python make_table.py               # -> results/tables.tex (Tables I and II)
python make_figure.py              # -> results/figure3.pdf (Scenario 2, K = 1024, realization 0)
```

* Run times are for an Apple M3 Max; other machines will differ. The proposed method
  needs well under a minute for all 2400 fits. The dsp methods (BTF and ABCO) need about
  18 CPU-hours in total (1.5 to 12.5 s per fit plus R start-up), i.e. roughly 2 hours
  with 8 to 12 parallel processes (`--jobs`, default: number of cores minus 2).
  `time_fits.py` runs one fit at a time and takes about 1.5 hours.
* Finished tasks are stored in `results/raw/` (and `results/timing.json`) and skipped on a
  re-run, so an interrupted run can simply be restarted. Results of an earlier version of
  the code must be moved out of `results/` first.
* Run times are taken from `time_fits.py`, not from `run_simulations.py`, whose parallel
  processes slow each other down. Without `results/timing.json`, `make_table.py` shows the
  run times recorded by `run_simulations.py` and marks them with `*`.

## What is computed

**Data** (`nupspline/signals.py`, Sec. IV). The signals Blocks, Bumps, HeaviSine and
Doppler [1] on `t_k = (k-1)/(K-1)`, scaled to standard deviation 7, with Gaussian noise of
variance 1 (Scenario 1) or 1 / 16 / 1 on `[0, 0.33) / [0.33, 0.66) / [0.66, 1]`
(Scenario 2). Realization `i` is drawn with
`numpy.random.default_rng([20240615, scenario - 1, signal index, K, i])`, so all methods
see exactly the same data.

**Methods** (Tables I and II). Competitors from the R package dsp with its default
settings (6000 iterations, 1000 burn-in), run on standardised data with a seed derived from
the data:

| Scenario | BTF [8], [9] | ABCO [10] (no outlier component) | Proposed |
|---|---|---|---|
| 1 | BTF-DHS | `m = 1` and `m = 2`, constant noise variance | constant noise variance: `gamma_{3,k}^2 = 0` |
| 2 | BTF-DHS-SV, BTF-DHS-ASV | `m = 1` and `m = 2`, ASV noise variance | piecewise-constant noise variance |

Table I shows ABCO with `m = 1`; Table II both.

**Standardization of the proposed method** (`smoother.fit`, `Config.standardize`). The
observations are centered by their median and divided by the noise level

    sigma_hat = 1.4826 * MAD(y~_2 - y~_1, ..., y~_K - y~_{K-1}) / sqrt(2),

the median absolute deviation (MAD) of the first differences (1.4826 = 1 / Phi^{-1}(3/4)
converts the MAD of Gaussian data into a standard deviation; sqrt(2) undoes the doubling of
the noise variance by differencing). After the fit, the estimates are transformed back.
All initial values are thus in units of `sigma_hat` and the method is invariant to the
location and scale of the data.

**Metrics.**
* *MSE* per realization, averaged over the realizations with its standard error (Table I).
* *Pooled F1 score* of the detected level jumps (Table II, `nupspline/jumps.py`). True
  jumps: Blocks (11) and HeaviSine (2); every detection on Bumps and Doppler is false.
  Proposed method: a jump at `k` if `gamma_{1,k}^2 > q_k^2`. ABCO: if the posterior
  changepoint probability is `>= c`, for the default `c = 0.5` and the best
  `c in {0.1, ..., 0.9}`. Two adjacent detections count as one; a detection within 5
  samples of a true jump is a hit, each true jump being matched at most once (pairing with
  the most hits). TP, FP and FN are summed over the four signals and all realizations
  before computing `F1 = 2 TP / (2 TP + FP + FN)`.
* *Run time* per fit on a single core, for the fit only: the proposed method after numba
  compilation, the dsp methods inside R around `dsp_fit` (without R start-up), averaged
  over 10 realizations x 4 signals (`time_fits.py`).

## From the paper to the code

| Paper | Code (`nupspline/`) |
|---|---|
| observations `y~_k`, estimate `yhat_k` (14) | `y`, `fit(...)["y_hat"]` |
| `X_{k+1} = A X_k + B U_k + B' U'_k`, `Y~_k = C X_k + Z_k` (10), (6) | `ssm.build_ssm(m)` |
| `m = 2` (cubic smoothing splines) | `Config.m` |
| `q_k^2`, `r_k^2` | `fit(...)["q2"]`, `fit(...)["r2"]` |
| `qbar_k = 1/q_k`, `rbar_k = 1/r_k` (Models 2 and 3) | `qbar`, `rbar` in `smoother._em` |
| NUV prior (8), update `gamma^2 = E[V^2]/beta` (9) | `g1sq` (Model 1), `gsq` in `_update_model23` |
| `beta1 = 3`, `beta2 = 1`, `beta3 = 2`, `alpha^2 = 0.8` | `Config.beta1/2/3`, `Config.alpha` |
| standardization by the median and `sigma_hat` | `Config.standardize`, `smoother.noise_level` |
| initial `qbar_k = rbar_k = 1`, `gamma_{i,k}^2 = 10^3` (in units of `sigma_hat`) | `Config.qbar_init`, `rbar_init`, `gamma_sq_init` |
| Scenario 1: `gamma_{3,k}^2 = 0` held fixed | `Config(constant_noise=True)` |
| E-step: MBF in Model 1 | `smoother._mbf_model1` |
| backward messages (16)-(18) / (20), (21) | first loop of `smoother._update_model23` |
| M-step (15) / (19): MBF in Model 2 / 3 (Fig. 2) | MBF loops of `smoother._update_model23` |
| 1000 EM iterations | `Config.n_iter` |
| detected level jumps `gamma_{1,k}^2 > q_k^2` | `detect_jumps(fit(...))` |
| ABCO threshold `c` | `jumps.ABCO_THRESHOLDS`, `abco.fit_abco(...)["cp_prob"]` |
| F1 score of Table II | `jumps.py`, `make_table.pooled_f1` |

Implementation details not stated in the paper (all in `config.py`, in units of
`sigma_hat`): the initial state of Model 1 has the prior `N(0, 10^3 I)`, and Models 2
and 3 start from `N(0, 10^6)`.

## Using the method on your own data

```python
import numpy as np
from nupspline import Config, detect_jumps, fit

y = ...                                   # 1-D array of noisy observations
est = fit(y, Config())                    # piecewise-constant noise variance
# est = fit(y, Config(constant_noise=True))   # constant noise variance
y_hat, r2, q2 = est["y_hat"], est["r2"], est["q2"]
jumps = detect_jumps(est)                 # indices of detected level jumps
```

The data are standardized internally (median and noise level), so no setting depends on
the units of `y`; `Config(standardize=False)` fits the raw data instead. The first call
compiles the code with numba (a few seconds; cached afterwards).

## Reproducibility

The proposed method is deterministic: it reproduces its entries of Tables I and II exactly.
The dsp fits use a seed derived from the data and are reproducible for a given R / dsp
installation; other platforms or versions may differ slightly in the last digits. Run
times depend on the hardware.

# Adaptive Smoothing of Time Series with Level Jumps and Time-Varying Noise via NUP Priors

Code for the simulations of the paper

> A. Lukaj and H.-A. Loeliger, *Adaptive Smoothing of Time Series with Level Jumps and
> Time-Varying Noise via NUP Priors.*

It reproduces **Table I** (average MSE and run time), **Table II** (F1 score of the
detected level jumps), **Fig. 2** (the proposed method on one realization of Scenario 2)
and **Fig. 3** (the competitors on the same realization).

## Contents

```
nupspline/                 the proposed method, the data, the competitors, the scoring
  config.py                parameters of the proposed method (Secs. II-E, IV-B)
  ssm.py                   state-space model (5)-(8), level-jump input B' (9)
  smoother.py              EM with MBF message passing, Steps 1-4 of Sec. III;
                           standardization of the data; level-jump detection
  signals.py               Donoho-Johnstone signals, Scenarios 1 and 2, seeds
  btf.py, dsp_btf.R        competitors BTF-DHS / -SV / -ASV [8], [9] via the R package dsp [22]
  abco.py, dsp_abco.R      competitor ABCO [10] via dsp
  jumps.py                 scoring of the detected level jumps (Table II)
run_simulations.py         runs the study (all methods, scenarios, signals, K)
time_fits.py               run times of Table I (one fit at a time)
make_table.py              builds Tables I and II (LaTeX + console)
make_figure.py             builds the figures (Figs. 2 and 3, or any methods)
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
python time_fits.py                # run times of Table I: 10 realizations x 4 signals per method and K, one at a time
python make_table.py               # -> results/tables.tex (Tables I and II)
python make_figure.py --paper      # -> results/figure2/figure2.tex (Fig. 2) and
                                   #    results/figure3/figure3.tex (Fig. 3)
                                   #    (Scenario 2, K = 1024, realization 0)
```

* Run times are for an Apple M3 Max; other machines will differ. The proposed method
  needs well under a minute for all 2400 fits. The dsp methods (BTF and ABCO) need about
  18 CPU-hours in total (1.5 to 12.5 s per fit plus R start-up), i.e. roughly 2 hours
  with 8 to 12 parallel processes (`--jobs`, default: number of cores minus 2).
  `time_fits.py` runs one fit at a time but only on the first 10 realizations
  (10 x 4 signals = 40 fits per method, scenario and K; `--realizations`), so it takes
  about 1.5 hours.
* Finished tasks are stored in `results/raw/` (and `results/timing.json`) and skipped on a
  re-run, so an interrupted run can simply be restarted. Results of an earlier version of
  the code must be moved out of `results/` first.
* Run times are taken from `time_fits.py`, not from `run_simulations.py`, whose parallel
  processes slow each other down. Without `results/timing.json`, `make_table.py` shows the
  run times recorded by `run_simulations.py` and marks them with `*`.

## Figures of single methods

`make_figure.py` plots any of the methods on one realization of Scenario 2 (four signals).
Each method gets three rows: the estimate with the detected level jumps, the estimated
`q_hat_k^2` and the true and estimated noise variance `r_k^2`, `r_hat_k^2`. For BTF and ABCO,
`q_hat_k^2` is the evolution variance of the `m`-th differences of the trend (`m = 2` for BTF).

```bash
python make_figure.py                                  # the proposed method -> results/figure_proposed.pdf (no R needed)
python make_figure.py --methods abco_m2                # ABCO-ASV with m = 2 -> results/figure_abco_m2.pdf
python make_figure.py --methods proposed btf abco_m1 --K 512 --realization 3 --out results/fig.pdf
python make_figure.py --methods abco_m1 --c 0.6        # ABCO: jumps where the changepoint probability >= 0.6
```

Methods: `proposed`, `btf` (BTF-DHS-ASV), `abco_m1`, `abco_m2` (ABCO-ASV). `--estimate-only`
keeps only the first row of each method; `--noisy-points N` plots only `N` of the noisy
samples. Figures with many panels may need lualatex instead of pdflatex (used automatically).

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
| 1 | BTF-DHS | `m = 1` and `m = 2`, constant noise variance | constant noise variance: `S_k = 0` held fixed |
| 2 | BTF-DHS-SV, BTF-DHS-ASV | `m = 1` and `m = 2`, ASV noise variance | piecewise-constant noise variance |

Tables I and II of the paper show ABCO with `m = 1`; `make_table.py` also prints ABCO with
`m = 2` in Table II.

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
  Proposed method: a jump at `k` if `sigma^2_{U'_k} > q_hat_k^2` (Sec. IV-B). ABCO: if the posterior
  changepoint probability is `>= c`, for the default `c = 0.5` and the best
  `c in {0.05, 0.10, ..., 0.95}`. Every flagged sample is a detection; a detection within 5
  samples of a true jump is a hit, each true jump being matched at most once (pairing with
  the most hits), so two detections near the same true jump give one TP and one FP. TP,
  FP and FN are summed over the four signals and all realizations before computing
  `F1 = 2 TP / (2 TP + FP + FN)`.
* *Run time* per fit on a single core, for the fit only: the proposed method after numba
  compilation, the dsp methods inside R around `dsp_fit` (without R start-up), averaged
  over 10 realizations x 4 signals (`time_fits.py`).

## From the paper to the code

Notation as in the paper (Fig. 1).

| Paper | Code (`nupspline/`) |
|---|---|
| observations `y~_k`, estimate `yhat_k = E[Y_k]` | `y`, `fit(...)["y_hat"]` |
| `X_{k+1} = A X_k + B U_k + B' U'_k`, `Y~_k = C X_k + Z_k` (9), (6); `A`, `a_i` (7), (8) | `ssm.build_ssm(m)` |
| `m = 2` (cubic smoothing splines) | `Config.m` |
| estimates `q_hat_k^2`, `r_hat_k^2` | `fit(...)["q2"]`, `fit(...)["r2"]` |
| `Q^inv_k = 1/Q_k`, `R^inv_k = 1/R_k`; `q_hat_k = 1/E[Q^inv_k]`, `r_hat_k = 1/E[R^inv_k]` | `qinv`, `rinv` in `smoother._em` |
| NUP prior (10) of the level jumps `U'_k`, variance `sigma^2_{U'_k}` | `s2Up` in `smoother._em`, `fit(...)["sigma2_Uprime"]` |
| `R^inv_{k+1} = R^inv_k + S_k` (12), variances `sigma^2_{S_k}`; the same for `Q^inv_k` with `S'_k`, `sigma^2_{S'_k}` | `s2S`, `s2Sp` in `smoother._em` |
| EM updates `sigma^2 = E[(.)^2] / beta` (11), (15) | Step 2 in `smoother._em`, Step 3c in `smoother._update_inv` |
| `beta = 3`, `beta_Q = 1`, `beta_R = 2`, `alpha^2 = 0.8` (Secs. II-E, IV-B) | `Config.beta`, `beta_Q`, `beta_R`, `alpha` |
| standardization by the median and `sigma_hat` (Sec. IV-A) | `Config.standardize`, `smoother.noise_level` |
| initial `r_hat_k = q_hat_k = 1`, `sigma^2_{U'_k} = sigma^2_{S'_k} = sigma^2_{S_k} = 10^3` (in units of `sigma_hat`) | `Config.rinv_init`, `qinv_init`, `sigma2_init` |
| Scenario 1: fixed `S_k = 0` | `Config(constant_noise=True)` |
| Step 1: Gaussian message passing (MBF) in (9), (6) | `smoother._mbf_signal` |
| Steps 3a / 4: backward messages (13), (14) | first loop of `smoother._update_inv` |
| Steps 3b, 3c / 4: MBF in (12) and update (15) | MBF loops and last loop of `smoother._update_inv` |
| 1000 EM iterations | `Config.n_iter` |
| detected level jumps `sigma^2_{U'_k} > q_hat_k^2` (Sec. IV-B) | `detect_jumps(fit(...))` |
| ABCO threshold `c` | `jumps.ABCO_THRESHOLDS`, `abco.fit_abco(...)["cp_prob"]` |
| F1 score of Table II | `jumps.py`, `make_table.pooled_f1` |

Implementation details not stated in the paper (all in `config.py`, in units of
`sigma_hat`): the initial states `X_0` and `R^inv_0`, `Q^inv_0` have the priors
`N(0, 10^3 I)` and `N(0, 10^3)`.

Indexing: array index `i` holds the paper's `k = i + 1` (`y[i] = y~_k`, `q2[i] = q_hat_k^2`,
`sigma2_Uprime[i] = sigma^2_{U'_k}`). The state-space model starts with `X_1 = A X_0`, without
an input; `U_k`, `U'_k` act between `y~_k` and `y~_{k+1}`, so `U_k = Delta^m Y_k` as in (2).
The inputs `U_k` with `k > K - m` would only affect values after `y~_K`, so they are set to
zero and send no backward message to `Q^inv_k` (`q_hat_k^2` for `k > K - m` continues
`q_hat_{K-m}^2`). The level jumps `U'_k` are restricted to the same range `k = 1, ..., K - m`,
as in Sec. II-C (`sigma^2_{U'_k} = 0` for `k > K - m`): a jump with `k > K - m` would only
shift the last `K - k < m` observations, which the inputs `U_k` can reproduce exactly, so it
would not be identifiable as a level jump.
`detect_jumps` returns indices `k` (0-based) of jumps between `y[k-1]` and `y[k]`.

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

# Adaptive Smoothing of Time Series with Level Jumps and Time-Varying Noise via NUP Priors

Code for the simulations of the paper

> A. Lukaj and H.-A. Loeliger, *Adaptive Smoothing of Time Series with Level Jumps and
> Time-Varying Noise via NUP Priors.*

It reproduces **Table I** (average MSE and run time) and **Fig. 3** (one realization of
Scenario 2) of the paper.

## Contents

```
nupspline/                 the proposed method and the data
  config.py                parameters of the proposed method (Sec. IV)
  ssm.py                   state-space model (5)-(7), level-jump input B' (10)
  smoother.py              EM with MBF message passing in Models 1, 2, 3 (Sec. III)
  signals.py               Donoho-Johnstone signals, Scenarios 1 and 2, seeds
  btf.py, dsp_btf.R        competitors BTF-DHS / -SV / -ASV via the R package dsp
run_simulations.py         runs the study (all methods, scenarios, signals, K)
make_table.py              builds Table I (LaTeX + console)
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

Without R, the proposed method can still be run
on its own (`--methods proposed`, `--no-btf`).

## Quick check (about 2 minutes)

```bash
python run_simulations.py --quick                # K = 128, 5 realizations, all methods
python make_table.py --results results_quick     # prints the table, writes table1.tex
python make_figure.py --out results_quick/figure3.pdf
```

Without R: `python run_simulations.py --quick --methods proposed` and
`python make_figure.py --no-btf`.

## Full reproduction of the paper

```bash
python run_simulations.py          # Table I: 2 scenarios x 4 signals x K in {128, 512, 1024} x 100 realizations
python make_table.py               # -> results/table1.tex
python make_figure.py              # -> results/figure3.pdf (Scenario 2, K = 1024, realization 0)
```

* All run times are for an Apple M3 Max; other machines will differ. The proposed method needs about one minute for all 1200 fits. The BTF methods need
  about 2.6 CPU-hours in total (5 to 14 s per fit), i.e. roughly 20 to 40 minutes with
  8 parallel processes (`--jobs`, default: number of cores minus 2).
* Finished tasks are stored in `results/raw/` and skipped on a re-run, so an interrupted
  run can simply be restarted.

## What is computed

**Data** (`nupspline/signals.py`, Sec. IV). The signals Blocks, Bumps, HeaviSine and
Doppler [1] on `t_k = (k-1)/(K-1)`, scaled to standard deviation 7, with Gaussian noise of
variance 1 (Scenario 1) or 1 / 16 / 1 on `[0, 0.33) / [0.33, 0.66) / [0.66, 1]`
(Scenario 2). Realization `i` is drawn with
`numpy.random.default_rng([20240615, scenario - 1, signal index, K, i])`, so all methods
see exactly the same data.

**Methods** (Table I):

| Scenario | Competitors (R package dsp, default settings: 6000 iterations, 1000 burn-in) | Proposed |
|---|---|---|
| 1 | BTF-DHS [8] | constant noise variance: `gamma_{3,k}^2 = 0` |
| 2 | BTF-DHS-SV [8], BTF-DHS-ASV [9] | piecewise-constant noise variance |

**Metrics.** MSE per realization as in (23), averaged over the realizations with its
standard error. Run time per fit on a single core, measured for the fit only: the
proposed method after numba compilation, the BTF methods inside R around `dsp_fit`
(without R start-up). The run times in the paper were measured on an Apple M3 Max.

## From the paper to the code

| Paper | Code (`nupspline/`) |
|---|---|
| observations `y~_k`, estimate `yhat_k` (15) | `y`, `fit(...)["y_hat"]` |
| `X_{k+1} = A X_k + B U_k + B' U'_k`, `Y~_k = C X_k + Z_k` (10), (6) | `ssm.build_ssm(m)` |
| `m = 2` (cubic smoothing splines) | `Config.m` |
| `q_k^2`, `r_k^2` | `fit(...)["q2"]`, `fit(...)["r2"]` |
| `qbar_k = 1/q_k`, `rbar_k = 1/r_k` (Models 2 and 3) | `qbar`, `rbar` in `smoother._em` |
| NUV prior (8), update `gamma^2 = E[V^2]/beta` (9) | `g1sq` (Model 1), `gsq` in `_update_model23` |
| `beta1 = 3`, `beta2 = 1`, `beta3 = 2`, `alpha^2 = 0.8` | `Config.beta1/2/3`, `Config.alpha` |
| initial `qbar_k = rbar_k = 0.5`, `gamma_{i,k}^2 = 10^3` | `Config.qbar_init`, `rbar_init`, `gamma_sq_init` |
| Scenario 1: `gamma_{3,k}^2 = 0` held fixed | `Config(constant_noise=True)` |
| E-step: MBF in Model 1 | `smoother._mbf_model1` |
| backward messages (18), (19) / (21), (22) | first loop of `smoother._update_model23` |
| M-step (16) / (20): MBF in Model 2 / 3 (Fig. 2) | MBF loops of `smoother._update_model23` |
| 1000 EM iterations | `Config.n_iter` |
| detected level jumps `gamma_{1,k}^2 > 1` | `fit(...)["gamma1_sq"] > 1` |

Implementation details not stated in the paper (all in `config.py`): the initial state of
Model 1 has the prior `N(0, 10^3 I)`, Models 2 and 3 start from `N(0, 10^6)`, and
`gamma_{2,k}^2`, `gamma_{3,k}^2` are capped at `10^6`.

## Using the method on your own data

```python
import numpy as np
from nupspline import Config, fit

y = ...                                   # 1-D array of noisy observations
est = fit(y, Config())                    # piecewise-constant noise variance
# est = fit(y, Config(constant_noise=True))   # constant noise variance
y_hat, r2, q2 = est["y_hat"], est["r2"], est["q2"]
jumps = np.flatnonzero(est["gamma1_sq"] > 1)  # indices of detected level jumps
```

The first call compiles the code with numba (a few seconds; cached afterwards).

## Reproducibility

The proposed method is deterministic: it reproduces Table I exactly. The BTF fits use a
seed derived from the data and are reproducible for a given R / dsp installation;
other platforms or versions may differ slightly in the last digits. Run times depend on
the hardware.

"""
Run the simulation study of Sec. IV (Tables I and II).

  Scenario 1 (noise variance 1):       BTF-DHS, ABCO m=1, ABCO m=2 (constant noise
                                       variance), proposed (S_k = 0)
  Scenario 2 (noise variance 1/16/1):  BTF-DHS-SV, BTF-DHS-ASV, ABCO-ASV m=1,
                                       ABCO-ASV m=2, proposed
  signals: Blocks, Bumps, HeaviSine, Doppler;  K = 128, 512, 1024;
  100 realizations each.

For every fit, the MSE and the run time are written to results/raw/, and for the
methods that detect level jumps also the per-sample jump score (proposed:
sigma^2_{U'_k} / q_hat_k^2; ABCO: posterior changepoint probability).  make_table.py
turns them into Tables I and II; the run-time rows of Table I come from
time_fits.py (one fit at a time, as in the paper).  Finished tasks are skipped, so
an interrupted run is resumed by starting it again.

Usage:
    python run_simulations.py --quick              # small check: K = 128, 5 realizations
    python run_simulations.py                      # the full study (Tables I and II)
    python run_simulations.py --methods proposed   # only the proposed method (no R needed)
    python run_simulations.py --jobs 8             # 8 parallel processes
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")      # one core per fit
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from nupspline import Config, SIGNALS, fit, jump_score, realization, warm_up

# method -> scenarios it is run in
METHODS = {"btf_dhs": [1], "btf_dhs_sv": [2], "btf_dhs_asv": [2],
           "abco_m1": [1], "abco_m2": [1], "abco_m1_asv": [2], "abco_m2_asv": [2],
           "proposed": [1, 2]}
# ABCO variants: method -> (m, obsSV)
ABCO = {"abco_m1": (1, "const"), "abco_m2": (2, "const"),
        "abco_m1_asv": (1, "ASV"), "abco_m2_asv": (2, "ASV")}
R_CHUNK = 25            # realizations per task for the (slow) dsp methods


def tasks(args):
    """Deterministic list of (method, scenario, signal, K, i0, i1)."""
    out = []
    for method in args.methods:
        for scenario in METHODS[method]:
            if scenario not in args.scenarios:
                continue
            for K in args.K:
                for signal in SIGNALS:
                    step = args.realizations if method == "proposed" else R_CHUNK
                    for i0 in range(0, args.realizations, step):
                        out.append((method, scenario, signal, K, i0,
                                    min(args.realizations, i0 + step)))
    return out


def result_file(out_dir, task):
    method, scenario, signal, K, i0, i1 = task
    return Path(out_dir) / "raw" / f"scenario{scenario}" / method / f"{signal}_K{K}_{i0:03d}-{i1 - 1:03d}.npz"


def fit_one(method, scenario, y):
    """Fit one realization; returns (y_hat, seconds, jump score or None)."""
    if method == "proposed":
        cfg = Config(constant_noise=(scenario == 1))
        t0 = time.perf_counter()
        est = fit(y, cfg)
        seconds = time.perf_counter() - t0
        return est["y_hat"], seconds, jump_score(est)                # jump where > 1
    if method in ABCO:
        from nupspline.abco import fit_abco
        m, obs_sv = ABCO[method]
        res = fit_abco(y, m, obs_sv)
        return res["y_hat"], res["seconds"], res["cp_prob"]         # jump where >= c
    from nupspline.btf import VARIANTS, fit_btf
    res = fit_btf(y, VARIANTS[method])
    return res["y_hat"], res["seconds"], None


def run_task(task, out_dir):
    out = result_file(out_dir, task)
    if out.exists():
        return f"skip {out.name}"
    method, scenario, signal, K, i0, i1 = task
    idx, mse, sec, score = [], [], [], []
    for i in range(i0, i1):
        t, y_obs, y_true, r_true = realization(signal, K, scenario, i)
        y_hat, seconds, s = fit_one(method, scenario, y_obs)
        idx.append(i)
        mse.append(float(np.mean((y_hat - y_true) ** 2)))
        sec.append(seconds)
        score.append(s)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    extra = {} if score[0] is None else {"jump_score": np.stack(score)}
    np.savez(tmp, realization=np.array(idx), mse=np.array(mse), seconds=np.array(sec), **extra)
    os.replace(tmp, out)
    return f"{method:12s} scenario {scenario}  {signal:9s} K={K:<5d} i={i0}..{i1 - 1}  " \
           f"mean MSE {np.mean(mse):.4f}  ({np.mean(sec):.3f} s/fit)"


def _worker_init():
    warm_up()             # compile the proposed method once per process


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    p.add_argument("--scenarios", nargs="+", type=int, default=[1, 2], choices=[1, 2])
    p.add_argument("--K", nargs="+", type=int, default=[128, 512, 1024])
    p.add_argument("--realizations", type=int, default=100)
    p.add_argument("--quick", action="store_true",
                   help="Small check run: K = 128, 5 realizations (results in results_quick/).")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                   help="Parallel processes (default: number of cores - 2).")
    p.add_argument("--out", type=Path, default=None, help="Output folder (default: results/).")
    args = p.parse_args()
    if args.quick:
        args.K, args.realizations = [128], 5
    out_dir = args.out or Path("results_quick" if args.quick else "results")

    todo = tasks(args)
    if any(m != "proposed" for m, *_ in todo):
        from nupspline.btf import check_dsp
        try:
            print(f"R package dsp {check_dsp()}")
        except RuntimeError as e:
            sys.exit(f"error: {e}\n(or run only the proposed method: --methods proposed)")

    print(f"{len(todo)} tasks, {args.jobs} in parallel -> {out_dir}/raw/")
    t0 = time.time()
    with ProcessPoolExecutor(args.jobs, initializer=_worker_init) as ex:
        futures = [ex.submit(run_task, t, out_dir) for t in todo]
        for n, f in enumerate(futures, 1):
            print(f"[{n}/{len(todo)}, {(time.time() - t0) / 60:.1f} min] {f.result()}", flush=True)
    print(f"done in {(time.time() - t0) / 60:.1f} min -- now: python make_table.py"
          + ("" if str(out_dir) == "results" else f" --results {out_dir}"))


if __name__ == "__main__":
    main()

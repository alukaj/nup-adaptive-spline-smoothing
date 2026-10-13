"""
Run the simulation study of Sec. IV (Table I).

  Scenario 1 (noise variance 1):        BTF-DHS, proposed (constant noise variance)
  Scenario 2 (noise variance 1/16/1):   BTF-DHS-SV, BTF-DHS-ASV, proposed
  signals: Blocks, Bumps, HeaviSine, Doppler;  K = 128, 512, 1024;
  100 realizations each.

Every fit's MSE (eq. (22)) and run time (fit only, single core: the proposed
method after compilation, the BTF methods timed inside R) is written to
results/raw/; make_table.py turns them into Table I.  Finished tasks are
skipped, so an interrupted run is resumed by starting it again.

Usage:
    python run_simulations.py --quick              # small check: K = 128, 5 realizations
    python run_simulations.py                      # the full study (Table I)
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

from nupspline import Config, SIGNALS, fit, realization, warm_up

# method -> scenarios it is run in (Table I)
METHODS = {"btf_dhs": [1], "btf_dhs_sv": [2], "btf_dhs_asv": [2], "proposed": [1, 2]}
BTF_CHUNK = 25          # realizations per task for the (slow) BTF methods


def tasks(args):
    """Deterministic list of (method, scenario, signal, K, i0, i1)."""
    out = []
    for method in args.methods:
        for scenario in METHODS[method]:
            if scenario not in args.scenarios:
                continue
            for K in args.K:
                for signal in SIGNALS:
                    step = args.realizations if method == "proposed" else BTF_CHUNK
                    for i0 in range(0, args.realizations, step):
                        out.append((method, scenario, signal, K, i0,
                                    min(args.realizations, i0 + step)))
    return out


def result_file(out_dir, task):
    method, scenario, signal, K, i0, i1 = task
    return Path(out_dir) / "raw" / f"scenario{scenario}" / method / f"{signal}_K{K}_{i0:03d}-{i1 - 1:03d}.npz"


def fit_one(method, scenario, y):
    """Fit one realization; returns (y_hat, seconds)."""
    if method == "proposed":
        cfg = Config(constant_noise=(scenario == 1))
        t0 = time.perf_counter()
        y_hat = fit(y, cfg)["y_hat"]
        return y_hat, time.perf_counter() - t0
    from nupspline.btf import VARIANTS, fit_btf
    res = fit_btf(y, VARIANTS[method])
    return res["y_hat"], res["seconds"]


def run_task(task, out_dir):
    out = result_file(out_dir, task)
    if out.exists():
        return f"skip {out.name}"
    method, scenario, signal, K, i0, i1 = task
    idx, mse, sec = [], [], []
    for i in range(i0, i1):
        t, y_obs, y_true, r_true = realization(signal, K, scenario, i)
        y_hat, seconds = fit_one(method, scenario, y_obs)
        idx.append(i)
        mse.append(float(np.mean((y_hat - y_true) ** 2)))
        sec.append(seconds)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, realization=np.array(idx), mse=np.array(mse), seconds=np.array(sec))
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

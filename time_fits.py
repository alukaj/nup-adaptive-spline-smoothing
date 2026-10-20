"""
Run times of Table I (bottom), measured as in the paper: one fit at a time on a
single core, for the fit only (the proposed method after compilation, the dsp
methods inside R around dsp_fit, without R start-up), averaged over
10 realizations x 4 signals = 40 fits per method, scenario and K.

Kept separate from run_simulations.py, which runs fits in parallel processes: run
times measured there are affected by the other processes.  Writes
results/timing.json, which make_table.py uses for the run-time rows.  Cells that are
already in the file are skipped, so an interrupted run is resumed by starting it again.

Usage:
    python time_fits.py                       # all Table I methods (about 1.5 hours)
    python time_fits.py --methods proposed    # only the proposed method (no R needed)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")      # one core per fit
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from nupspline import SIGNALS, realization, warm_up
from run_simulations import fit_one

# the methods with a run-time column in Table I, with their scenario
TABLE1 = [("btf_dhs", 1), ("abco_m1", 1), ("proposed", 1),
          ("btf_dhs_sv", 2), ("btf_dhs_asv", 2), ("abco_m1_asv", 2), ("proposed", 2)]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=sorted({m for m, _ in TABLE1}),
                   choices=sorted({m for m, _ in TABLE1}))
    p.add_argument("--K", nargs="+", type=int, default=[128, 512, 1024])
    p.add_argument("--realizations", type=int, default=10)
    p.add_argument("--out", type=Path, default=Path("results") / "timing.json")
    args = p.parse_args()

    cells = [(m, s, K) for m, s in TABLE1 if m in args.methods for K in args.K]
    if any(m != "proposed" for m, _, _ in cells):
        from nupspline.btf import check_dsp
        try:
            print(f"R package dsp {check_dsp()}")
        except RuntimeError as e:
            sys.exit(f"error: {e}\n(or time only the proposed method: --methods proposed)")
    warm_up()
    timing = json.loads(args.out.read_text()) if args.out.exists() else {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for method, scenario, K in cells:
        key = f"{method}/scenario{scenario}/K{K}"
        if key in timing:
            print(f"skip {key}")
            continue
        sec = [fit_one(method, scenario, realization(s, K, scenario, i)[1])[1]
               for s in SIGNALS for i in range(args.realizations)]
        timing[key] = sec
        args.out.write_text(json.dumps(timing, indent=1))
        print(f"{key:28s} mean {np.mean(sec):.4g} s  (SE {np.std(sec, ddof=1) / np.sqrt(len(sec)):.2g}, "
              f"n = {len(sec)})", flush=True)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

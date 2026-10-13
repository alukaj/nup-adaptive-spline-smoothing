"""
Build Table I of the paper from the results of run_simulations.py.

Top: average MSE over the realizations (eq. (23)) with its standard error in
units of the last digit, e.g. 0.335(7) = 0.335 +- 0.007; best per row and
scenario in bold.  Bottom: average run time per fit in seconds.

Writes <results>/table1.tex (compiles on its own) and prints the table.

Usage:
    python make_table.py                       # results/
    python make_table.py --results results_quick
"""
import argparse
from pathlib import Path

import numpy as np

from nupspline import SIGNALS, TITLES

COLUMNS = [(1, "btf_dhs", "DHS"), (1, "proposed", "Prop."),
           (2, "btf_dhs_sv", "DHS-SV"), (2, "btf_dhs_asv", "DHS-ASV"), (2, "proposed", "Prop.")]


def load(results):
    """{(scenario, method, signal, K): (mse array, seconds array)}"""
    data = {}
    for f in sorted(Path(results, "raw").glob("scenario*/*/*.npz")):
        scenario, method = int(f.parent.parent.name[len("scenario"):]), f.parent.name
        signal, rest = f.stem.split("_K")
        K = int(rest.split("_")[0])
        z = np.load(f)
        mse, sec = data.get((scenario, method, signal, K), (np.empty(0), np.empty(0)))
        data[(scenario, method, signal, K)] = (np.r_[mse, z["mse"]], np.r_[sec, z["seconds"]])
    return data


def fmt_mse(x):
    """mean(SE in units of the last digit): 3 decimals while the SE has <= 2 digits there."""
    mean, se = float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
    for d in (3, 2, 1, 0):
        if round(se * 10 ** d) < 100 or d == 0:
            return f"{mean:.{d}f}({int(round(se * 10 ** d))})"


def fmt_sec(x):
    s = float(np.mean(x))
    return f"{s:.3f}" if s < 0.1 else (f"{s:.2f}" if s < 10 else f"{s:.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    args = p.parse_args()
    data = load(args.results)
    if not data:
        raise SystemExit(f"no results in {args.results}/raw -- run run_simulations.py first")
    Ks = sorted({k[3] for k in data})
    n_real = sorted({len(v[0]) for v in data.values()})

    rows_tex, rows_txt = [], []
    for K in Ks:
        rows_tex.append(r"\midrule")
        for j, signal in enumerate(SIGNALS):
            cells, plain = [], []
            for scen in (1, 2):
                cols = [(m, (scen, m, signal, K)) for s, m, _ in COLUMNS if s == scen]
                means = [np.mean(data[k][0]) if k in data else np.inf for _, k in cols]
                best = int(np.argmin(means)) if np.isfinite(min(means)) else -1
                for c, (_, k) in enumerate(cols):
                    txt = fmt_mse(data[k][0]) if k in data else "--"
                    cells.append(rf"\textbf{{{txt}}}" if c == best else txt)
                    plain.append(("*" if c == best else "") + txt)
            lead = rf"\multirow{{4}}{{*}}{{{K}}}" if j == 0 else ""
            rows_tex.append(f"{lead} & {TITLES[signal]} & " + " & ".join(cells) + r" \\")
            rows_txt.append(f"{K:>5} {TITLES[signal]:10s} " + " ".join(f"{c:>12s}" for c in plain))
    time_tex, time_txt = [], []
    for K in Ks:
        cells = []
        for scen, m, _ in COLUMNS:
            secs = [data[(scen, m, s, K)][1] for s in SIGNALS if (scen, m, s, K) in data]
            cells.append(fmt_sec(np.concatenate(secs)) if secs else "--")
        time_tex.append(f"{K} & & " + " & ".join(cells) + r" \\")
        time_txt.append(f"{K:>5} {'':10s} " + " ".join(f"{c:>12s}" for c in cells))

    head = " & ".join(h for _, _, h in COLUMNS)
    nr = n_real[0] if len(n_real) == 1 else f"{min(n_real)}--{max(n_real)}"
    tex = rf"""\documentclass{{article}}
\usepackage{{booktabs,multirow}}
\begin{{document}}
\begin{{table}}[t]
\centering
\caption{{Top: average MSE over {nr} realizations, with standard errors in units of the
last digit in parentheses; best per row and scenario in bold. Bottom: average run time
per fit in seconds.}}
\begin{{tabular}}{{ll cc ccc}}
\toprule
 & & \multicolumn{{2}}{{c}}{{Scenario 1}} & \multicolumn{{3}}{{c}}{{Scenario 2}} \\
\cmidrule(lr){{3-4}} \cmidrule(lr){{5-7}}
$K$ & Signal & {head} \\
{chr(10).join(rows_tex)}
\midrule
\multicolumn{{7}}{{l}}{{\textit{{Run time [s]}}}} \\
\midrule
{chr(10).join(time_tex)}
\bottomrule
\end{{tabular}}
\end{{table}}
\end{{document}}
"""
    out = args.results / "table1.tex"
    out.write_text(tex)
    print(f"{'K':>5} {'Signal':10s} " + " ".join(f"{h:>12s}" for _, _, h in COLUMNS))
    print(f"{'':16s} {'--- Scenario 1 ---':>25s} {'--------- Scenario 2 ---------':>38s}")
    print("\n".join(rows_txt))
    print("Run time per fit [s]")
    print("\n".join(time_txt))
    print(f"\n(* = best per row and scenario)  LaTeX: {out}")


if __name__ == "__main__":
    main()

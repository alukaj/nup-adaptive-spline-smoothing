"""
Build Tables I and II of the paper from the results of run_simulations.py (and the
run times of time_fits.py).

Table I.   Top: average MSE over the realizations with its standard error in units of
           the last digit, e.g. 0.335(7) = 0.335 +- 0.007; best per row and scenario in
           bold.  Bottom: average run time per fit in seconds, from results/timing.json
           (time_fits.py); without it, the run times recorded by run_simulations.py are
           shown instead (marked: measured with parallel processes).
Table II.  Pooled F1 score of the detected level jumps (nupspline/jumps.py): proposed
           method (gamma_{1,k}^2 > q_k^2) and ABCO with m = 1, 2 at its default
           threshold c = 0.5 and at the best c in {0.05, 0.10, ..., 0.95}: one c per ABCO
           variant, the one with the largest F1 pooled over both scenarios, all K and all
           signals (as the proposed method uses one rule throughout); best per column in bold.

Writes <results>/tables.tex (compiles on its own) and prints both tables.

Usage:
    python make_table.py                       # results/
    python make_table.py --results results_quick
"""
import argparse
import json
from pathlib import Path

import numpy as np

from nupspline import SIGNALS, TITLES
from nupspline.jumps import ABCO_THRESHOLDS, counts, detect, f1, true_jumps

# Table I: (scenario, method, header)
COLUMNS = [(1, "btf_dhs", "DHS"), (1, "abco_m1", "ABCO"), (1, "proposed", "Prop."),
           (2, "btf_dhs_sv", "SV"), (2, "btf_dhs_asv", "ASV"), (2, "abco_m1_asv", "ASV"),
           (2, "proposed", "Prop.")]
# Table II: (label, {scenario: method}, threshold or None for the best c)
F1_ROWS = [("Proposed", {1: "proposed", 2: "proposed"}, None),
           ("ABCO $m{=}1$, $c = 0.5$", {1: "abco_m1", 2: "abco_m1_asv"}, 0.5),
           ("ABCO $m{=}1$, best $c$", {1: "abco_m1", 2: "abco_m1_asv"}, "best"),
           ("ABCO $m{=}2$, $c = 0.5$", {1: "abco_m2", 2: "abco_m2_asv"}, 0.5),
           ("ABCO $m{=}2$, best $c$", {1: "abco_m2", 2: "abco_m2_asv"}, "best")]


def load(results):
    """{(scenario, method, signal, K): dict(mse, seconds, jump_score or None)},
    sorted by realization."""
    parts = {}
    for f in sorted(Path(results, "raw").glob("scenario*/*/*.npz")):
        scenario, method = int(f.parent.parent.name[len("scenario"):]), f.parent.name
        signal, rest = f.stem.split("_K")
        K = int(rest.split("_")[0])
        parts.setdefault((scenario, method, signal, K), []).append(np.load(f))
    data = {}
    for key, zs in parts.items():
        order = np.argsort(np.concatenate([z["realization"] for z in zs]))
        score = (np.concatenate([z["jump_score"] for z in zs])[order]
                 if all("jump_score" in z.files for z in zs) else None)
        data[key] = {"mse": np.concatenate([z["mse"] for z in zs])[order],
                     "seconds": np.concatenate([z["seconds"] for z in zs])[order],
                     "jump_score": score}
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


def pooled_counts(data, scenario, method, K, thr):
    """(TP, FP, FN) summed over the four signals and all realizations, or None if missing."""
    tp = fp = fn = 0
    for signal in SIGNALS:
        d = data.get((scenario, method, signal, K))
        if d is None or d["jump_score"] is None:
            return None
        true = true_jumps(signal, K)
        for s in d["jump_score"]:
            det = detect(s, 1.0, strict=True) if method == "proposed" else detect(s, thr)
            a, b, c = counts(det, true)
            tp, fp, fn = tp + a, fp + b, fn + c
    return tp, fp, fn


def pooled_f1(data, scenario, method, K, thr):
    """Pooled F1 over the four signals and all realizations, or None if missing."""
    n = pooled_counts(data, scenario, method, K, thr)
    return None if n is None else f1(*n)


def best_threshold(data, methods, Ks):
    """The c in ABCO_THRESHOLDS with the largest F1 pooled over both scenarios, all K and
    all signals (the smaller c on ties), or None without results."""
    res = []
    for c in ABCO_THRESHOLDS:
        n = [pooled_counts(data, scen, methods[scen], K, c) for scen in (1, 2) for K in Ks]
        n = [x for x in n if x is not None]
        if n:
            res.append((f1(*np.sum(n, axis=0)), c))
    return max(res, key=lambda t: (t[0], -t[1]))[1] if res else None


def table1(data, timing, Ks):
    rows_tex, rows_txt = [], []
    for K in Ks:
        rows_tex.append(r"\midrule")
        for j, signal in enumerate(SIGNALS):
            cells, plain = [], []
            for scen in (1, 2):
                keys = [(scen, m, signal, K) for s, m, _ in COLUMNS if s == scen]
                means = [np.mean(data[k]["mse"]) if k in data else np.inf for k in keys]
                best = int(np.argmin(means)) if np.isfinite(min(means)) else -1
                for c, k in enumerate(keys):
                    txt = fmt_mse(data[k]["mse"]) if k in data else "--"
                    cells.append(rf"\textbf{{{txt}}}" if c == best else txt)
                    plain.append(("*" if c == best else "") + txt)
            lead = rf"\multirow{{4}}{{*}}{{{K}}}" if j == 0 else ""
            rows_tex.append(f"{lead} & {TITLES[signal]} & " + " & ".join(cells) + r" \\")
            rows_txt.append(f"{K:>5} {TITLES[signal]:10s} " + " ".join(f"{c:>11s}" for c in plain))
    time_tex, time_txt, parallel = [], [], False
    for K in Ks:
        cells = []
        for scen, m, _ in COLUMNS:
            key = f"{m}/scenario{scen}/K{K}"
            if key in timing:
                cells.append(fmt_sec(timing[key]))
                continue
            secs = [data[(scen, m, s, K)]["seconds"] for s in SIGNALS if (scen, m, s, K) in data]
            cells.append(fmt_sec(np.concatenate(secs)) + "*" if secs else "--")
            parallel = parallel or bool(secs)
        time_tex.append(f"{K} & & " + " & ".join(cells) + r" \\")
        time_txt.append(f"{K:>5} {'':10s} " + " ".join(f"{c:>11s}" for c in cells))
    return rows_tex, rows_txt, time_tex, time_txt, parallel


def table2(data, Ks):
    """Rows of (label, [cell per (scenario, K)]), value matrix for the bold marks."""
    rows, vals = [], []
    for label, methods, thr in F1_ROWS:
        missing = False
        if thr == "best":
            thr = best_threshold(data, methods, Ks)
            missing = thr is None
            if not missing:
                label = label.replace("best $c$", f"best $c = {thr:g}$")
        cells, v = [], []
        for scen in (1, 2):
            for K in Ks:
                x = None if missing else pooled_f1(data, scen, methods[scen], K, thr)
                cells.append(None if x is None else (x, None)); v.append(np.nan if x is None else x)
        rows.append((label, cells)); vals.append(v)
    return rows, np.array(vals, float)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, default=Path("results"))
    args = p.parse_args()
    data = load(args.results)
    if not data:
        raise SystemExit(f"no results in {args.results}/raw -- run run_simulations.py first")
    tfile = args.results / "timing.json"
    timing = json.loads(tfile.read_text()) if tfile.exists() else {}
    Ks = sorted({k[3] for k in data})
    n_real = sorted({len(v["mse"]) for v in data.values()})
    nr = n_real[0] if len(n_real) == 1 else f"{min(n_real)}--{max(n_real)}"

    rows_tex, rows_txt, time_tex, time_txt, parallel = table1(data, timing, Ks)
    f1_rows, f1_vals = table2(data, Ks)
    best = np.nanmax(np.where(np.isnan(f1_vals), -np.inf, f1_vals), axis=0)
    f1_tex, f1_txt = [], []
    for r, (label, cells) in enumerate(f1_rows):
        tex, txt = [], []
        for j, cell in enumerate(cells):
            if cell is None:
                tex.append("--"); txt.append("--"); continue
            x, c = cell
            s = f"{x:.2f}"
            bold = np.isfinite(best[j]) and round(x, 2) == round(best[j], 2)
            t = rf"\textbf{{{s}}}" if bold else s
            tex.append(t + (f"$_{{({c:g})}}$" if c is not None else ""))
            txt.append(("*" if bold else "") + s + (f"({c:g})" if c is not None else ""))
        f1_tex.append(f"{label} & " + " & ".join(tex) + r" \\")
        if r in (0, 2):
            f1_tex.append(r"\midrule")
        f1_txt.append(f"{label.replace('$', '').replace('{=}', '='):22s} " + " ".join(f"{t:>11s}" for t in txt))

    nK = len(Ks)
    cols2 = "c" * nK
    k_heads = " & ".join(str(K) for K in Ks)
    rows1, times1, rows2 = "\n".join(rows_tex), "\n".join(time_tex), "\n".join(f1_tex)
    time_note = (" Run times marked * were recorded during run\\_simulations.py with parallel "
                 "processes; run time\\_fits.py for the protocol of the paper." if parallel else "")
    tex = rf"""\documentclass{{article}}
\usepackage{{booktabs,multirow}}
\usepackage[margin=2cm]{{geometry}}
\begin{{document}}
\begin{{table}}[t]
\centering
\caption{{Top: average MSE over {nr} realizations, with standard errors in units of the last
digit in parentheses; best per row and scenario in bold. ABCO with $m = 1$. Bottom: average
run time per fit in seconds.{time_note}}}
\begin{{tabular}}{{ll ccc cccc}}
\toprule
 & & \multicolumn{{3}}{{c}}{{Scenario 1}} & \multicolumn{{4}}{{c}}{{Scenario 2}} \\
\cmidrule(lr){{3-5}} \cmidrule(lr){{6-9}}
 & & & & & \multicolumn{{2}}{{c}}{{DHS}} & ABCO & \\
\cmidrule(lr){{6-7}} \cmidrule(lr){{8-8}}
$K$ & Signal & DHS & ABCO & Prop. & SV & ASV & ASV & Prop. \\
{rows1}
\midrule
\multicolumn{{9}}{{l}}{{\textit{{Run time [s]}}}} \\
\midrule
{times1}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{table}}[t]
\centering
\caption{{Pooled $F_1$ score of the detected level jumps (hit: detection within 5 samples of a
true jump, each matched at most once). For ABCO, best $c$: the single threshold in
$\{{0.05, 0.10, \dots, 0.95\}}$ with the largest $F_1$ pooled over both scenarios, all $K$ and
all signals. Best per column in bold.}}
\begin{{tabular}}{{l {cols2} {cols2}}}
\toprule
 & \multicolumn{{{nK}}}{{c}}{{Scenario 1}} & \multicolumn{{{nK}}}{{c}}{{Scenario 2}} \\
\cmidrule(lr){{2-{1 + nK}}} \cmidrule(lr){{{2 + nK}-{1 + 2 * nK}}}
$K$ & {k_heads} & {k_heads} \\
\midrule
{rows2}
\bottomrule
\end{{tabular}}
\end{{table}}
\end{{document}}
"""
    out = args.results / "tables.tex"
    out.write_text(tex)
    heads = [f"{h}" for _, _, h in COLUMNS]
    print("TABLE I")
    print(f"{'K':>5} {'Signal':10s} " + " ".join(f"{h:>11s}" for h in heads))
    print(f"{'':16s} {'------ Scenario 1 ------':>35s} {'------------ Scenario 2 ------------':>47s}")
    print("\n".join(rows_txt))
    print("Run time per fit [s]" + ("   (* = recorded with parallel processes; run time_fits.py)"
                                    if parallel else ""))
    print("\n".join(time_txt))
    print("\nTABLE II (pooled F1)")
    print(f"{'':22s} " + " ".join(f"{f'S{s} K={K}':>11s}" for s in (1, 2) for K in Ks))
    print("\n".join(f1_txt))
    print(f"\n(* = best per row and scenario / per column)  LaTeX: {out}")


if __name__ == "__main__":
    main()

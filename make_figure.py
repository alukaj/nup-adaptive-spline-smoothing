"""
Figures of one realization of Scenario 2, all four signals.

Each method gets three rows: the noisy data, the ground truth and the estimate
yhat_k with the detected level jumps as red dots; the estimated q_k^2; the true
and estimated noise variance r_k^2.  Methods (--methods, in this order, on the
same data):
  proposed  the proposed method; jumps where sigma^2_{U'_k} > q_hat_k^2
  btf       BTF-DHS-ASV [9]; no jump detection
  abco_m1   ABCO-ASV [10] with m = 1, and abco_m2 with m = 2; jumps where the
            posterior changepoint probability is >= c (dsp's default c = 0.5)
For BTF and ABCO, "q_k^2" is the evolution variance of the m-th differences of the
trend (m = 2 for BTF).  BTF and ABCO need R + dsp.  --estimate-only keeps only the
first row of each method, with a single legend.

The figures of the paper (--paper, K = 1024, realization 0):
  results/figure2            Fig. 2: the proposed method
  results/figure3            Fig. 3: BTF-DHS-ASV, ABCO-ASV m = 1 and m = 2, estimate only

Each figure is written twice: with matplotlib (--out, a quick look) and as a
pgfplots figure (<out without suffix>/<name>.tex with its tables in data/, same
axis limits), compiled with pdflatex (or lualatex) if available.

Usage:
    python make_figure.py --paper                      # both figures of the paper
    python make_figure.py                              # -> results/figure_proposed.pdf (no R needed)
    python make_figure.py --methods abco_m2            # -> results/figure_abco_m2.pdf
    python make_figure.py --methods proposed btf --K 512 --realization 3 --out results/fig.pdf
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nupspline import SIGNALS, TITLES, Config, detect_jumps, fit, realization  # noqa: E402
from nupspline.jumps import detect  # noqa: E402

# pgfplots styles of the paper figure, and the legend of each row of a method
NOISY = "only marks, mark=*, mark size=0.3pt, color=gray!55"
TRUTH = "black, line width=0.5pt"
EST = "blue, line width=0.5pt"
JUMP = "only marks, mark=*, mark size=1.3pt, red"
Q2 = "green!50!black, line width=0.5pt"
R2_TRUE = "black, dashed, line width=0.5pt"
R2_EST = "orange!90!black, line width=0.5pt"
LEGENDS = ([("only marks, mark=*, mark size=1.5pt, color=gray!55", r"noisy $\breve y_k$"),
            (TRUTH, "ground truth"), (EST, r"estimated $\hat y_k$"), (JUMP, "detected jump")],
           [(Q2, r"estimated $\hat q_k^2$")],
           [(R2_TRUE, r"true $r_k^2$"), (R2_EST, r"estimated $\hat r_k^2$")])


def draw(axes, t, y_obs, y_true, r_true, est, jumps=None, show_legend=False, jump_label=False,
         shown=slice(None)):
    """The rows of one method: y_k, and (if three axes) q_k^2 and r_k^2.  jump_label:
    give the legend a "detected jump" entry even if the method has no jumps; shown:
    the noisy samples plotted."""
    ax_y = axes[0]
    ax_y.plot(t[shown], y_obs[shown], ".", color="0.7", ms=2, label=r"noisy $\breve y_k$")
    ax_y.plot(t, y_true, "k-", lw=0.8, label="ground truth")
    ax_y.plot(t, est["y_hat"], "b-", lw=1.0, label=r"estimated $\hat y_k$")
    if jumps is not None or jump_label:      # (an empty plot still gives the legend entry)
        idx = [] if jumps is None else jumps
        ax_y.plot(t[idx], est["y_hat"][idx], "o", color="red", ms=4, label="detected jump")
    if len(axes) == 3:
        ax_q, ax_r = axes[1:]
        ax_q.semilogy(t, est["q2"], color="green", lw=1.0, label=r"estimated $\hat q_k^2$")
        lo, hi = np.nanmin(est["q2"]), np.nanmax(est["q2"])
        if hi / lo < 100:          # nearly constant: show at least two decades
            mid = np.sqrt(lo * hi)
            ax_q.set_ylim(mid / 10, mid * 10)
        ax_r.plot(t, r_true ** 2, "k--", lw=1.0, label=r"true $r_k^2$")
        ax_r.plot(t, est["r2"], color="darkorange", lw=1.0, label=r"estimated $\hat r_k^2$")
    if show_legend:
        for ax in axes:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)


def tex_series(t, y_obs, y_true, r_true, est, jumps=None, n_kinds=3, shown=slice(None)):
    """The curves of the first n_kinds rows of one method, as lists of (style, x, y);
    shown: the noisy samples plotted."""
    row_y = [(NOISY, t[shown], y_obs[shown]), (TRUTH, t, y_true), (EST, t, est["y_hat"])]
    if jumps is not None and len(jumps):
        row_y.append((JUMP, t[jumps], est["y_hat"][jumps]))
    if n_kinds == 1:
        return [row_y]
    return [row_y, [(Q2, t, est["q2"])], [(R2_TRUE, t, r_true ** 2), (R2_EST, t, est["r2"])]]


def write_tex(path, series, ylims, row_labels, with_jumps, n_kinds=3):
    """pgfplots figure: series[col][row] = list of (style, x, y), ylims[row][col] =
    (ymin, ymax), row_labels = y label of each method (n_kinds rows each, None: none),
    with_jumps = whether each method shows detected jumps; n_kinds = 3 (rows y_k,
    q_k^2, r_k^2 per method, a legend for each row) or 1 (row y_k only, one legend on
    the first row); tables in path.parent/data/."""
    data_dir = path.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(ylims)
    legend_rows = range(n_rows) if n_kinds == 3 else [0]
    plots = []
    for r in range(n_rows):
        kind, method = r % n_kinds, r // n_kinds       # kind 0: y_k, 1: q_k^2, 2: r_k^2
        jump_legend = with_jumps[method] if n_kinds == 3 else any(with_jumps)
        for c, signal in enumerate(SIGNALS):
            opts = [r"xlabel={$k/K$}" if r == n_rows - 1 else "xticklabels={}"]
            if kind == 1:
                opts.append("ymode=log")
            lo, hi = ylims[r][c]
            opts += [f"ymin={lo:g}", f"ymax={hi:g}"]
            if kind == 0:
                opts += ["height=3.0cm", "width=4.0cm"]
            if c == 0 and kind == 0 and row_labels[method]:
                # two short lines, e.g. "ABCO-ASV \\ (m=1)", so that the labels fit the rows
                label = row_labels[method].replace(" (", r"\\(").replace("DHS-ASV", r"DHS-\\ASV")
                opts += [f"ylabel={{{label}}}", r"ylabel style={align=center, font=\footnotesize}"]
            legend_here = c == len(SIGNALS) - 1 and r in legend_rows
            if legend_here:
                opts.append(f"legend to name=LegendRow{r + 1}")
            lines = [rf"  \nextgroupplot[{', '.join(opts)}]"]
            for j, (style, x, y) in enumerate(series[c][r], 1):
                name = f"data/{signal}_row{r + 1}_{j}.dat"
                ok = np.isfinite(y)                    # ABCO: no q_k^2 for the first m samples
                np.savetxt(path.parent / name, np.column_stack([x[ok], y[ok]]), fmt="%.6g")
                lines.append(rf"    \addplot[{style}, forget plot] table {{{name}}};")
            if legend_here:
                legend = LEGENDS[kind] if kind or jump_legend else LEGENDS[0][:3]
                for style, text in legend:
                    lines += [rf"    \addlegendimage{{{style}}}", rf"    \addlegendentry{{{text}}}"]
            plots.append("\n".join(lines))
    vsep = "0.15cm" if n_kinds == 3 else "0.4cm"    # n_kinds = 1: room for the method labels
    # beside its row; a single legend (n_kinds = 1) beside the middle of all rows
    last = len(SIGNALS)
    at = {r: f"(group c{last}r{r + 1}.east)" for r in legend_rows}
    if n_kinds == 1:
        at[0] = f"(group c{last}r1.east)!0.5!(group c{last}r{n_rows}.east)"
    nodes = "\n".join(rf"\node[anchor=west, text width=2.5cm] at ($" + at[r] +
                      rf"+(2mm,0)$) {{\pgfplotslegendfromname{{LegendRow{r + 1}}}}};"
                      for r in legend_rows)
    path.write_text(rf"""\documentclass{{standalone}}
\usepackage{{amsmath,amssymb}}
\usepackage{{pgfplots}}
\usepgfplotslibrary{{groupplots}}
\usetikzlibrary{{calc}}
\pgfplotsset{{compat=1.18}}
\begin{{document}}
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{group name=group, group size={len(SIGNALS)} by {n_rows}, horizontal sep=1.15cm, vertical sep={vsep}}},
    width=4cm, height=2.4cm,
    xmin=0, xmax=1, xtick={{0,0.2,...,1}},
    scaled y ticks=false, tick label style={{font=\scriptsize}},
    title style={{font=\small}}, label style={{font=\scriptsize}},
    tick align=outside, tick pos=left, clip mode=individual,
    legend cell align=left,
    legend style={{font=\scriptsize, draw=none, fill=none, inner sep=0pt, row sep=1pt}},
  ]
{chr(10).join(plots)}
\end{{groupplot}}
{nodes}
\end{{tikzpicture}}
\end{{document}}
""")


LABELS = {"proposed": "proposed", "btf": "BTF-DHS-ASV",
          "abco_m1": "ABCO-ASV ($m=1$)", "abco_m2": "ABCO-ASV ($m=2$)"}

# --paper: (file name in the results folder, methods, estimate only, noisy samples shown)
PAPER_FIGURES = [
    ("figure2.pdf", ["proposed"], False, None),                             # Fig. 2
    ("figure3.pdf", ["btf", "abco_m1", "abco_m2"], True, 500),   # Fig. 3; 500: fits pdflatex
]


def fit_method(method, y_obs, c):
    """Estimate (y_hat, q2, r2) and detected jumps (None: the method has none)."""
    if method == "proposed":
        est = fit(y_obs, Config(constant_noise=False))
        return est, detect_jumps(est)
    if method == "btf":
        from nupspline.btf import fit_btf
        return fit_btf(y_obs, "ASV"), None
    from nupspline.abco import fit_abco
    est = fit_abco(y_obs, int(method[-1]), "ASV")
    return est, detect(est["cp_prob"], c)


def make_figure(methods, out, K=1024, realization_idx=0, c=0.5, estimate_only=False,
                noisy_points=None):
    """One figure: matplotlib (out) and pgfplots (<out without suffix>/<name>.tex);
    noisy_points: number of evenly spaced noisy samples plotted (None: all)."""
    n_kinds = 1 if estimate_only else 3
    shown = (slice(None) if noisy_points is None or noisy_points >= K else
             np.unique(np.linspace(0, K - 1, noisy_points).round().astype(int)))
    n_rows = n_kinds * len(methods)
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, (2.0 if n_kinds == 3 else 2.6) * n_rows),
                             sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1, 1][:n_kinds] * len(methods)})
    series = [[] for _ in SIGNALS]
    with_jumps = [m != "btf" for m in methods]
    for i, method in enumerate(methods):
        rows = slice(n_kinds * i, n_kinds * (i + 1))
        for col, signal in enumerate(SIGNALS):
            t, y_obs, y_true, r_true = realization(signal, K, 2, realization_idx)
            est, jumps = fit_method(method, y_obs, c)
            legend = col == 3 and (n_kinds == 3 or i == 0)
            draw(axes[rows, col], t, y_obs, y_true, r_true, est, jumps=jumps, show_legend=legend,
                 jump_label=n_kinds == 1 and any(with_jumps), shown=shown)
            series[col] += tex_series(t, y_obs, y_true, r_true, est, jumps, n_kinds, shown)
            mse = np.mean((est["y_hat"] - y_true) ** 2)
            n_jumps = "" if jumps is None else f"  {len(jumps)} jumps"
            print(f"{TITLES[signal]:10s} {LABELS[method]:20s} MSE {mse:.4f}{n_jumps}")
        axes[n_kinds * i, 0].set_ylabel(LABELS[method])
    for col, signal in enumerate(SIGNALS):
        axes[0, col].set_title(TITLES[signal])
        axes[-1, col].set_xlabel("$k/K$")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"figure -> {out}")

    # the same figure in pgfplots, with the axis limits of the matplotlib figure
    tex = out.with_suffix("") / f"{out.stem}.tex"
    tex.parent.mkdir(parents=True, exist_ok=True)
    ylims = [[axes[r, j].get_ylim() for j in range(len(SIGNALS))] for r in range(n_rows)]
    plt.close(fig)
    row_labels = [None] if methods == ["proposed"] else [LABELS[m] for m in methods]
    write_tex(tex, series, ylims, row_labels, with_jumps, n_kinds)
    print(f"LaTeX  -> {tex}")
    # lualatex if pdflatex fails: many scatter plots exceed pdflatex's fixed main memory
    for engine in (e for e in ("pdflatex", "lualatex") if shutil.which(e)):
        r = subprocess.run([engine, "-interaction=nonstopmode", tex.name],
                           cwd=tex.parent, capture_output=True, text=True)
        if r.returncode == 0:
            print(f"         {tex.with_suffix('.pdf')} ({engine})")
            break
        print(f"{engine} failed, see {tex.with_suffix('.log')}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paper", action="store_true",
                   help="Make the figures of the paper (PAPER_FIGURES) in the folder of --out "
                        "(needs R + dsp).")
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--realization", type=int, default=0)
    p.add_argument("--methods", nargs="+", choices=list(LABELS),
                   default=["proposed"], help="Methods to show, in this order (default: proposed).")
    p.add_argument("--estimate-only", action="store_true",
                   help="Show only the row y_k of each method (no q_k^2, r_k^2).")
    p.add_argument("--noisy-points", type=int, default=None,
                   help="Plot only this many evenly spaced noisy samples (default: all K); "
                        "the fits use all K.")
    p.add_argument("--c", type=float, default=0.5,
                   help="ABCO: jump where the posterior changepoint probability is >= c (default 0.5, dsp's).")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PDF (default: results/figure_<methods>.pdf; "
                        "with --paper, its folder is used, default results/).")
    args = p.parse_args()
    common = dict(K=args.K, realization_idx=args.realization, c=args.c)
    if args.paper:
        folder = args.out.parent if args.out else Path("results")
        for name, methods, estimate_only, noisy_points in PAPER_FIGURES:
            print(f"--- {name}")
            make_figure(methods, folder / name, estimate_only=estimate_only,
                        noisy_points=noisy_points, **common)
        return
    methods = list(dict.fromkeys(args.methods))
    out = args.out or Path("results") / f"figure_{'_'.join(methods)}.pdf"
    make_figure(methods, out, estimate_only=args.estimate_only,
                noisy_points=args.noisy_points, **common)


if __name__ == "__main__":
    main()

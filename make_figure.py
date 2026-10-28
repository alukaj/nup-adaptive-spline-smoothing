"""
Fig. 3 of the paper: one realization of Scenario 2 with K = 1024, all four
signals.  Top three rows: proposed method (noisy data, ground truth, estimate
yhat_k with detected level jumps gamma_{1,k}^2 > q_k^2 as red dots; estimated
q_k^2; true and estimated r_k^2).  With --with-btf, three more rows show
BTF-DHS-ASV [9] on the same data (needs R + dsp).

Writes the figure twice: with matplotlib (--out, a quick look) and as the pgfplots
figure of the paper (<out without suffix>/figure3.tex with its tables in data/, same
axis limits), compiled with pdflatex if available.

Usage:
    python make_figure.py                 # -> results/figure3.pdf, results/figure3/figure3.tex
    python make_figure.py --with-btf --realization 3 --K 512
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


def draw(axes, t, y_obs, y_true, r_true, est, jumps=None, show_legend=False):
    ax_y, ax_q, ax_r = axes
    ax_y.plot(t, y_obs, ".", color="0.7", ms=2, label=r"noisy $\breve y_k$")
    ax_y.plot(t, y_true, "k-", lw=0.8, label="ground truth")
    ax_y.plot(t, est["y_hat"], "b-", lw=1.0, label=r"estimated $\hat y_k$")
    if jumps is not None:      # (an empty plot still gives the legend entry)
        ax_y.plot(t[jumps], est["y_hat"][jumps], "o", color="red", ms=4, label="detected jump")
    ax_q.semilogy(t, est["q2"], color="green", lw=1.0, label=r"estimated $\hat q_k^2$")
    lo, hi = np.min(est["q2"]), np.max(est["q2"])
    if hi / lo < 100:          # nearly constant: show at least two decades
        mid = np.sqrt(lo * hi)
        ax_q.set_ylim(mid / 10, mid * 10)
    ax_r.plot(t, r_true ** 2, "k--", lw=1.0, label=r"true $r_k^2$")
    ax_r.plot(t, est["r2"], color="darkorange", lw=1.0, label=r"estimated $\hat r_k^2$")
    ax_r.set_xlabel("$t$")
    if show_legend:
        for ax in axes:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)


def tex_series(t, y_obs, y_true, r_true, est, jumps=None):
    """The curves of the three rows of one method, as lists of (style, x, y)."""
    row_y = [(NOISY, t, y_obs), (TRUTH, t, y_true), (EST, t, est["y_hat"])]
    if jumps is not None and len(jumps):
        row_y.append((JUMP, t[jumps], est["y_hat"][jumps]))
    return [row_y, [(Q2, t, est["q2"])], [(R2_TRUE, t, r_true ** 2), (R2_EST, t, est["r2"])]]


def write_tex(path, series, ylims, row_labels):
    """pgfplots figure: series[col][row] = list of (style, x, y), ylims[row][col] =
    (ymin, ymax), row_labels = y label of each method (three rows each, None: none);
    tables in path.parent/data/."""
    data_dir = path.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    n_rows = len(ylims)
    plots = []
    for r in range(n_rows):
        kind = r % 3                                   # 0: y_k, 1: q_k^2, 2: r_k^2
        for c, signal in enumerate(SIGNALS):
            opts = [r"xlabel={$t$}" if r == n_rows - 1 else "xticklabels={}"]
            if kind == 1:
                opts.append("ymode=log")
            lo, hi = ylims[r][c]
            opts += [f"ymin={lo:g}", f"ymax={hi:g}"]
            if kind == 0:
                opts += ["height=3.0cm", "width=4.0cm"]
            if c == 0 and kind == 0 and row_labels[r // 3]:
                opts.append(f"ylabel={{{row_labels[r // 3]}}}")
            if c == len(SIGNALS) - 1:
                opts.append(f"legend to name=LegendRow{r + 1}")
            lines = [rf"  \nextgroupplot[{', '.join(opts)}]"]
            for j, (style, x, y) in enumerate(series[c][r], 1):
                name = f"data/{signal}_row{r + 1}_{j}.dat"
                np.savetxt(path.parent / name, np.column_stack([x, y]), fmt="%.6g")
                lines.append(rf"    \addplot[{style}, forget plot] table {{{name}}};")
            if c == len(SIGNALS) - 1:
                legend = LEGENDS[kind] if kind or r == 0 else LEGENDS[0][:3]   # jumps: proposed only
                for style, text in legend:
                    lines += [rf"    \addlegendimage{{{style}}}", rf"    \addlegendentry{{{text}}}"]
            plots.append("\n".join(lines))
    nodes = "\n".join(rf"\node[anchor=west, text width=2.5cm] at ($(group c{len(SIGNALS)}r{r + 1}.east)"
                      rf"+(2mm,0)$) {{\pgfplotslegendfromname{{LegendRow{r + 1}}}}};"
                      for r in range(n_rows))
    path.write_text(rf"""\documentclass[border=3pt]{{standalone}}
\usepackage{{amsmath,amssymb}}
\usepackage{{pgfplots}}
\usepgfplotslibrary{{groupplots}}
\usetikzlibrary{{calc}}
\pgfplotsset{{compat=1.18}}
\begin{{document}}
\begin{{tikzpicture}}
\begin{{groupplot}}[
    group style={{group name=group, group size={len(SIGNALS)} by {n_rows}, horizontal sep=1.15cm, vertical sep=0.15cm}},
    width=4cm, height=2.4cm,
    xmin=0, xmax=1, xtick={{0,0.2,...,1}},
    scaled y ticks=false, tick label style={{font=\scriptsize}},
    title style={{font=\small}}, label style={{font=\small}},
    axis lines=box, tick align=outside, tick pos=left, clip mode=individual,
    legend cell align=left,
    legend style={{font=\scriptsize, draw=none, fill=none, inner sep=0pt, /tikz/every row/.append style={{row sep=1pt}}}},
  ]
{chr(10).join(plots)}
\end{{groupplot}}
{nodes}
\end{{tikzpicture}}
\end{{document}}
""")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--realization", type=int, default=0)
    p.add_argument("--with-btf", action="store_true",
                   help="Add BTF-DHS-ASV on the same data (needs R + dsp).")
    p.add_argument("--out", type=Path, default=Path("results") / "figure3.pdf")
    args = p.parse_args()

    n_rows = 6 if args.with_btf else 3
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 2.0 * n_rows), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1] * (n_rows // 3)})
    series = []
    for col, signal in enumerate(SIGNALS):
        t, y_obs, y_true, r_true = realization(signal, args.K, 2, args.realization)
        est = fit(y_obs, Config(constant_noise=False))
        jumps = detect_jumps(est)
        draw(axes[0:3, col], t, y_obs, y_true, r_true, est, jumps=jumps, show_legend=(col == 3))
        series.append(tex_series(t, y_obs, y_true, r_true, est, jumps))
        axes[0, col].set_title(TITLES[signal])
        mse = np.mean((est["y_hat"] - y_true) ** 2)
        print(f"{TITLES[signal]:10s} proposed     MSE {mse:.4f}")
        if args.with_btf:
            from nupspline.btf import fit_btf
            btf = fit_btf(y_obs, "ASV")
            draw(axes[3:6, col], t, y_obs, y_true, r_true, btf, show_legend=(col == 3))
            series[-1] += tex_series(t, y_obs, y_true, r_true, btf)
            print(f"{TITLES[signal]:10s} BTF-DHS-ASV  MSE {np.mean((btf['y_hat'] - y_true) ** 2):.4f}")
    axes[0, 0].set_ylabel("proposed")
    if args.with_btf:
        axes[3, 0].set_ylabel("BTF-DHS-ASV")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"figure -> {args.out}")

    # the same figure in pgfplots, with the axis limits of the matplotlib figure
    tex = args.out.with_suffix("") / "figure3.tex"
    tex.parent.mkdir(parents=True, exist_ok=True)
    ylims = [[axes[r, c].get_ylim() for c in range(len(SIGNALS))] for r in range(n_rows)]
    write_tex(tex, series, ylims, ["proposed", "BTF-DHS-ASV"] if args.with_btf else [None])
    print(f"LaTeX  -> {tex}")
    if shutil.which("pdflatex"):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", tex.name],
                           cwd=tex.parent, capture_output=True, text=True)
        print(f"         {tex.with_suffix('.pdf')}" if r.returncode == 0 else
              f"pdflatex failed, see {tex.with_suffix('.log')}")


if __name__ == "__main__":
    main()

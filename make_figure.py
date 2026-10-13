"""
Fig. 3 of the paper: one realization of Scenario 2 with K = 1024, all four
signals.  Top three rows: proposed method (noisy data, ground truth, estimate
yhat_k with detected level jumps gamma_{1,k}^2 > 1 as red dots; estimated
q_k^2; true and estimated r_k^2).  Bottom three rows: BTF-DHS-ASV [9]
(skipped with --no-btf, e.g. without R).

Usage:
    python make_figure.py                 # -> results/figure3.pdf
    python make_figure.py --no-btf --realization 3 --K 512
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from nupspline import SIGNALS, TITLES, Config, fit, realization  # noqa: E402


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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--K", type=int, default=1024)
    p.add_argument("--realization", type=int, default=0)
    p.add_argument("--no-btf", action="store_true", help="Only the proposed method.")
    p.add_argument("--out", type=Path, default=Path("results") / "figure3.pdf")
    args = p.parse_args()

    n_rows = 3 if args.no_btf else 6
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 2.0 * n_rows), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1] * (n_rows // 3)})
    for col, signal in enumerate(SIGNALS):
        t, y_obs, y_true, r_true = realization(signal, args.K, 2, args.realization)
        est = fit(y_obs, Config(constant_noise=False))
        draw(axes[0:3, col], t, y_obs, y_true, r_true, est,
             jumps=est["gamma1_sq"] > 1.0, show_legend=(col == 3))
        axes[0, col].set_title(TITLES[signal])
        mse = np.mean((est["y_hat"] - y_true) ** 2)
        print(f"{TITLES[signal]:10s} proposed     MSE {mse:.4f}")
        if not args.no_btf:
            from nupspline.btf import fit_btf
            btf = fit_btf(y_obs, "ASV")
            draw(axes[3:6, col], t, y_obs, y_true, r_true, btf, show_legend=(col == 3))
            print(f"{TITLES[signal]:10s} BTF-DHS-ASV  MSE {np.mean((btf['y_hat'] - y_true) ** 2):.4f}")
    axes[0, 0].set_ylabel("proposed")
    if not args.no_btf:
        axes[3, 0].set_ylabel("BTF-DHS-ASV")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"figure -> {args.out}")


if __name__ == "__main__":
    main()

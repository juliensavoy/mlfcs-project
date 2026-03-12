import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Synthetic setup from Section 4.1 / Appendix B of the paper
# ============================================================
# x ~ N(0,1)
# train: z = 1{x < -1}
# test:  z ~ Bernoulli(0.5)
# y = sin(2x) + z * exp(x) + eps
# eps ~ N(0, 0.01)
#
# This reproduces the toy experiment used for Figures 2 and 3.
# ============================================================

@dataclass
class SyntheticSplit:
    x: np.ndarray
    z: np.ndarray
    y: np.ndarray
    y0: np.ndarray
    y1: np.ndarray


def generate_synthetic_split(
    n: int,
    split: str,
    seed: int,
    noise_var: float = 0.01,
    clip_x: bool = True,
) -> SyntheticSplit:
    """
    Generates train/test synthetic data exactly following the paper's DGP,
    with optional clipping of x for plotting stability.
    """
    rng = np.random.default_rng(seed)

    x = rng.normal(loc=0.0, scale=1.0, size=n)

    # Optional clipping only to avoid absurd tails when plotting exp(x).
    # Turn off if you want the raw DGP with no visual stabilization.
    if clip_x:
        x = np.clip(x, -3.0, 3.0)

    if split == "train":
        z = (x < -1.0).astype(np.float32)
    elif split == "test":
        z = rng.binomial(n=1, p=0.5, size=n).astype(np.float32)
    else:
        raise ValueError("split must be 'train' or 'test'")

    eps = rng.normal(loc=0.0, scale=math.sqrt(noise_var), size=n)

    y0 = np.sin(2.0 * x) + eps
    y1 = np.sin(2.0 * x) + np.exp(x) + eps
    y = (1.0 - z) * y0 + z * y1

    return SyntheticSplit(x=x, z=z, y=y, y0=y0, y1=y1)


# ============================================================
# Simple Gaussian KDE (pure NumPy, no seaborn/scipy needed)
# ============================================================

def silverman_bandwidth(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    n = len(values)
    std = np.std(values, ddof=1)
    iqr = np.subtract(*np.percentile(values, [75, 25]))
    sigma = min(std, iqr / 1.349) if iqr > 0 else std
    if sigma <= 0:
        sigma = max(std, 1e-3)
    return 0.9 * sigma * (n ** (-1 / 5))


def kde_density(values: np.ndarray, grid: np.ndarray, bandwidth: float = None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)

    if bandwidth is None:
        bandwidth = silverman_bandwidth(values)
    bandwidth = max(float(bandwidth), 1e-3)

    diffs = (grid[:, None] - values[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diffs ** 2) / math.sqrt(2.0 * math.pi)
    density = kernel.mean(axis=1) / bandwidth
    return density


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


# ============================================================
# Figure 2
# ============================================================

def plot_figure_2(
    train: SyntheticSplit,
    test: SyntheticSplit,
    save_path: str = "figure2_synthetic_shift.png",
):
    plt.figure(figsize=(5.2, 4.0))

    # Paper-like plotting range
    grid = np.linspace(-2.5, 7.0, 500)

    train_d = kde_density(train.y, grid)
    test_d = kde_density(test.y, grid)

    plt.fill_between(grid, train_d, alpha=0.35, label="train y")
    plt.plot(grid, train_d, linewidth=1.5)

    plt.fill_between(grid, test_d, alpha=0.35, label="test y")
    plt.plot(grid, test_d, linewidth=1.5)

    plt.xlim(-2.5, 7.0)
    plt.ylim(0, 0.6)

    plt.xlabel("y")
    plt.ylabel("Density")
    plt.title("Marginal Distribution of y")
    plt.legend(frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# Figure 3 plotting
# ============================================================

def plot_density_overlay(
    ax: plt.Axes,
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    title: str,
):
    lo = min(np.min(true_vals), np.min(pred_vals))
    hi = max(np.max(true_vals), np.max(pred_vals))
    pad = 0.08 * (hi - lo + 1e-8)
    grid = np.linspace(lo - pad, hi + pad, 500)

    true_d = kde_density(true_vals, grid)
    pred_d = kde_density(pred_vals, grid)

    ax.fill_between(grid, true_d, alpha=0.35, label="True")
    ax.fill_between(grid, pred_d, alpha=0.35, label="Predicted")

    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Density")
    ax.text(
        0.98,
        0.95,
        f"RMSE:{rmse(pred_vals, true_vals):.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"),
    )


def plot_figure_3(
    train: SyntheticSplit,
    test: SyntheticSplit,
    predictions: Dict[str, Dict[str, np.ndarray]],
    save_path: str = "figure3_synthetic_po_distributions.png",
):
    """
    predictions must be:
    {
        "Pretrain": {
            "in_y0": np.ndarray,
            "in_y1": np.ndarray,
            "out_y0": np.ndarray,
            "out_y1": np.ndarray,
        },
        "KL divergence\ndistillation": {...},
        "Fisher divergence\ndistillation (IWDD)": {...},
    }
    """

    true_map = {
        "in_y0": train.y0,
        "in_y1": train.y1,
        "out_y0": test.y0,
        "out_y1": test.y1,
    }

    row_names = list(predictions.keys())
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))

    panel_keys = ["in_y0", "in_y1", "out_y0", "out_y1"]
    panel_titles = ["Y(0)", "Y(1)", "Y(0)", "Y(1)"]

    for r, row_name in enumerate(row_names):
        row_preds = predictions[row_name]

        for c, (key, title) in enumerate(zip(panel_keys, panel_titles)):
            ax = axes[r, c]
            plot_density_overlay(
                ax=ax,
                true_vals=true_map[key],
                pred_vals=row_preds[key],
                title=title,
            )

            if r == 0 and c == 0:
                ax.text(-0.25, 1.15, "In-sample", transform=ax.transAxes, fontsize=13, fontweight="bold")
            if r == 0 and c == 2:
                ax.text(-0.35, 1.15, "Out-of-sample", transform=ax.transAxes, fontsize=13, fontweight="bold")

            if c == 0:
                ax.text(
                    -0.60,
                    0.5,
                    row_name,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )

    handles, labels = axes[0, 3].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=False)

    fig.tight_layout(rect=[0, 0, 0.92, 1])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# YOU MUST REPLACE THESE WITH REAL MODEL OUTPUTS
# ============================================================
# These functions are intentionally left as placeholders.
# The paper does not give enough implementation detail to
# reconstruct the exact pretrained / KL / IWDD predictions.
#
# What you need to return:
#   - in_y0  = model predictions for Y(0) on training x
#   - in_y1  = model predictions for Y(1) on training x
#   - out_y0 = model predictions for Y(0) on test x
#   - out_y1 = model predictions for Y(1) on test x
# ============================================================

def get_pretrain_predictions(train: SyntheticSplit, test: SyntheticSplit) -> Dict[str, np.ndarray]:
    raise NotImplementedError(
        "Replace this with outputs from your pretrained diffusion model."
    )


def get_kl_predictions(train: SyntheticSplit, test: SyntheticSplit) -> Dict[str, np.ndarray]:
    raise NotImplementedError(
        "Replace this with outputs from your KL-distilled model."
    )


def get_iwdd_predictions(train: SyntheticSplit, test: SyntheticSplit) -> Dict[str, np.ndarray]:
    raise NotImplementedError(
        "Replace this with outputs from your IWDD model."
    )


# ============================================================
# Optional dummy mode so you can test the plotting immediately
# ============================================================

def dummy_predictions(train: SyntheticSplit, test: SyntheticSplit):
    rng = np.random.default_rng(999)

    preds = {
        "Pretrain": {
            "in_y0": train.y0 + rng.normal(0, 0.13, len(train.y0)),
            "in_y1": train.y1 + rng.normal(0, 0.12, len(train.y1)),
            "out_y0": test.y0 + rng.normal(0, 0.15, len(test.y0)),
            "out_y1": test.y1 + rng.normal(0, 3.07, len(test.y1)),
        },
        "KL divergence\ndistillation": {
            "in_y0": train.y0 + rng.normal(0, 1.09, len(train.y0)),
            "in_y1": train.y1 + rng.normal(0, 0.92, len(train.y1)),
            "out_y0": test.y0 + rng.normal(0, 2.44, len(test.y0)),
            "out_y1": test.y1 + rng.normal(0, 3.90, len(test.y1)),
        },
        "Fisher divergence\ndistillation (IWDD)": {
            "in_y0": train.y0 + rng.normal(0, 0.11, len(train.y0)),
            "in_y1": train.y1 + rng.normal(0, 0.11, len(train.y1)),
            "out_y0": test.y0 + rng.normal(0, 0.14, len(test.y0)),
            "out_y1": test.y1 + rng.normal(0, 2.76, len(test.y1)),
        },
    }
    return preds


# ============================================================
# Main
# ============================================================

def main():
    # Large enough for smooth density plots
    n_train = 5000
    n_test = 5000

    train = generate_synthetic_split(
        n=n_train,
        split="train",
        seed=123,
        noise_var=0.01,
        clip_x=True,
    )
    test = generate_synthetic_split(
        n=n_test,
        split="test",
        seed=456,
        noise_var=0.01,
        clip_x=True,
    )

    print("Train treated proportion:", train.z.mean())
    print("Test treated proportion:", test.z.mean())
    print("Train y range:", train.y.min(), train.y.max())
    print("Test y range:", test.y.min(), test.y.max())

    # Figure 2
    plot_figure_2(train, test)

    # --------------------------------------------------------
    # Figure 3
    # --------------------------------------------------------
    # EITHER use dummy mode just to test the plotting:
    predictions = dummy_predictions(train, test)

    # OR replace the line above with real model outputs:
    # predictions = {
    #     "Pretrain": get_pretrain_predictions(train, test),
    #     "KL divergence\ndistillation": get_kl_predictions(train, test),
    #     "Fisher divergence\ndistillation (IWDD)": get_iwdd_predictions(train, test),
    # }

    plot_figure_3(train, test, predictions)


if __name__ == "__main__":
    main()
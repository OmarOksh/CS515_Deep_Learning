"""
Compact plotting script for ablation studies.

Each experiment group gets ONE subplot with both training (dashed)
and validation (solid) curves overlaid. Produces two compact figures:
    1. combined_accuracies.png  (7 subplots in a 4x2 grid)
    2. combined_losses.png      (7 subplots in a 4x2 grid)
"""

import os
import re
import matplotlib.pyplot as plt


def parse_log_file(filepath):
    """
    Extract per-epoch train/val loss and accuracy from a log file.

    Args:
        filepath (str): Path to log .txt file.

    Returns:
        dict with keys: train_loss, val_loss, train_acc, val_acc.
    """
    data = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    with open(filepath, "r") as f:
        for line in f:
            t = re.search(r"Train loss: ([\d.]+)\s+acc: ([\d.]+)", line)
            if t:
                data["train_loss"].append(float(t.group(1)))
                data["train_acc"].append(float(t.group(2)))
            v = re.search(r"Val\s+loss: ([\d.]+)\s+acc: ([\d.]+)", line)
            if v:
                data["val_loss"].append(float(v.group(1)))
                data["val_acc"].append(float(v.group(2)))
    return data


# ============================================================
# Experiment groups
# ============================================================
GROUPS = {
    "Architecture": {
        "experiments": ["baseline", "arch_shallow", "arch_deep", "arch_narrow", "arch_wide"],
        "labels": {
            "baseline": "Baseline (512, 256)",
            "arch_shallow": "Shallow (512)",
            "arch_deep": "Deep (512, 256, 128, 64)",
            "arch_narrow": "Narrow (64, 32)",
            "arch_wide": "Wide (1024, 512)",
        },
    },
    "Dropout": {
        "experiments": ["baseline", "drop_0.0", "drop_0.7"],
        "labels": {
            "baseline": "Baseline (0.3)",
            "drop_0.0": "Dropout (0.0)",
            "drop_0.7": "Dropout (0.7)",
        },
    },
    "L2 Regularization": {
        "experiments": ["baseline", "l2_0.0", "l2_1e-2"],
        "labels": {
            "baseline": "Baseline (1e-4)",
            "l2_0.0": "L2 (0.0)",
            "l2_1e-2": "L2 (0.01)",
        },
    },
    "L1 Regularization": {
        "experiments": ["baseline", "l1_1e-5", "l1_1e-4", "l1_1e-3"],
        "labels": {
            "baseline": "Baseline (L1=0)",
            "l1_1e-5": "L1 (1e-5)",
            "l1_1e-4": "L1 (1e-4)",
            "l1_1e-3": "L1 (1e-3)",
        },
    },
    "Activation": {
        "experiments": ["baseline", "act_gelu"],
        "labels": {
            "baseline": "ReLU (Baseline)",
            "act_gelu": "GELU",
        },
    },
    "Batch Normalization": {
        "experiments": ["baseline", "bn_after_act", "bn_disabled"],
        "labels": {
            "baseline": "BN Before Act (Baseline)",
            "bn_after_act": "BN After Act",
            "bn_disabled": "No BN",
        },
    },
    "LR Scheduler": {
        "experiments": ["baseline", "sched_cosine", "sched_plateau"],
        "labels": {
            "baseline": "StepLR (Baseline)",
            "sched_cosine": "CosineAnnealing",
            "sched_plateau": "ReduceLROnPlateau",
        },
    },
}

# Distinct colors per experiment within a group
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def plot_combined(metric_key, ylabel, title_word, save_name, log_dir="logs"):
    """
    One subplot per group. Train = dashed, Val = solid, same color per experiment.

    Args:
        metric_key: 'acc' or 'loss'.
        ylabel: Y-axis label.
        title_word: Used in the figure suptitle.
        save_name: Output filename.
        log_dir: Directory containing log .txt files.
    """
    n = len(GROUPS)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.8 * nrows))
    fig.suptitle(f"Ablation Studies — {title_word}", fontsize=17, y=0.995)
    axes_flat = axes.flatten()

    for i, (group_name, cfg) in enumerate(GROUPS.items()):
        ax = axes_flat[i]

        for j, exp in enumerate(cfg["experiments"]):
            filepath = os.path.join(log_dir, f"{exp}.txt")
            if not os.path.exists(filepath):
                print(f"  [SKIP] {filepath}")
                continue

            data = parse_log_file(filepath)
            train_vals = data[f"train_{metric_key}"]
            val_vals = data[f"val_{metric_key}"]
            epochs = range(1, len(train_vals) + 1)
            label = cfg["labels"].get(exp, exp)
            color = COLORS[j % len(COLORS)]

            # Train = dashed (no legend entry), Val = solid (legend entry)
            ax.plot(epochs, train_vals, linestyle="--", color=color, alpha=0.5,
                    linewidth=1.2)
            ax.plot(epochs, val_vals, linestyle="-", marker="o", markersize=3,
                    color=color, linewidth=1.8, label=label)

        ax.set_title(group_name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epochs", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7.5, loc="best")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)

        # Add a small annotation for the line style legend
        ax.annotate("solid = val, dashed = train", xy=(0.98, 0.02),
                    xycoords="axes fraction", fontsize=6.5, ha="right",
                    color="grey", style="italic")

    # Hide unused subplot(s)
    for k in range(n, len(axes_flat)):
        axes_flat[k].set_visible(False)

    plt.tight_layout(rect=[0, 0.0, 1, 0.97])
    os.makedirs("plots", exist_ok=True)
    path = os.path.join("plots", save_name)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    plot_combined("acc", "Accuracy",
                  "Training (dashed) vs Validation (solid) Accuracy",
                  "combined_accuracies.png")
    plot_combined("loss", "Loss",
                  "Training (dashed) vs Validation (solid) Loss",
                  "combined_losses.png")
    print("\nDone! Check plots/ folder.")
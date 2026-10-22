"""plot accuracy on pairs.csv (differences) vs pairs_pairwise.csv (pairwise), per model.

One panel per model (eval_output/<model>/eval_results.csv with a matching
pairwise_eval_results.csv beside it, normative and CHILDES-formatted):
a row per phenomenon plus overall critical and overall control, with the
differences and pairwise accuracies joined by a line.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

from scripts.plot_eval import CHILDES_PREFIX, INK, MODEL_DISPLAY_NAMES, MUTED, style_axes

VERSION_COLORS = {"differences": "#2a78d6", "pairwise": "#eb6834"}


def load_runs(eval_dir):
    """[(label, differences results, pairwise results)] for every model with both."""
    runs = []
    for prefix in ("", CHILDES_PREFIX):
        for path in sorted(glob.glob(os.path.join(eval_dir, "*", f"{prefix}eval_results.csv"))):
            model = os.path.basename(os.path.dirname(path))
            pairwise_path = os.path.join(os.path.dirname(path), f"{prefix}pairwise_eval_results.csv")
            if not os.path.exists(pairwise_path):
                continue
            label = MODEL_DISPLAY_NAMES.get(model, model) + (" (CHILDES)" if prefix else "")
            read = lambda p: pd.read_csv(p, keep_default_na=False, na_values=[])
            runs.append((label, read(path), read(pairwise_path)))
    # normative logprob, then SLOR, then CHILDES-formatted SLOR
    runs.sort(key=lambda run: ("CHILDES" in run[0], "slor" in run[0]))
    if not runs:
        raise SystemExit(f"No <model>/eval_results.csv with a matching pairwise_ file in {eval_dir}")
    return runs


def accuracy_rows(results):
    """Accuracy per phenomenon (critical), then overall critical and overall control."""
    critical = results[results["condition_type"] != "control"]
    control = results[results["condition_type"] == "control"]
    acc = critical.groupby("phenomenon")["correct"].mean()
    acc["Overall (critical)"] = critical["correct"].mean()
    acc["Overall (control)"] = control["correct"].mean()
    return acc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", default="eval_output")
    parser.add_argument("--output", default="figures/plot_pairwise_vs_differences.png")
    args = parser.parse_args()

    runs = load_runs(args.eval_dir)
    fig, axes = plt.subplots(1, len(runs), figsize=(3.2 * len(runs) + 1.5, 4.2), sharey=True, sharex=True)
    axes = [axes] if len(runs) == 1 else axes
    for ax, (label, differences, pairwise) in zip(axes, runs):
        table = pd.DataFrame({"differences": accuracy_rows(differences), "pairwise": accuracy_rows(pairwise)})
        order = [r for r in table.index if not r.startswith("Overall")] + ["Overall (critical)", "Overall (control)"]
        table = table.loc[order[::-1]]
        y = range(len(table))
        ax.hlines(y, table.min(axis=1), table.max(axis=1), color=MUTED, linewidth=1.5, alpha=0.5, zorder=1)
        for version, color in VERSION_COLORS.items():
            ax.scatter(table[version], y, s=50, color=color, edgecolor="white", linewidth=1.5, zorder=2, label=version)
        ax.axhline(1.5, color=MUTED, linewidth=0.8, alpha=0.4)
        ax.set_yticks(list(y), table.index)
        ax.set_title(label, fontsize=10, color=INK)
        ax.set_xlim(0.5, 1.02)
        ax.set_xticks([0.6, 0.8, 1.0])
        ax.set_xlabel("accuracy", color=MUTED)
        style_axes(ax)
    axes[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Accuracy: differences of differences vs pairwise minimal pairs", color=INK)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

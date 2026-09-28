"""plot eval results (from eval.py) across every model found in eval_output/.

Three figures:
  1. accuracy by phenomenon: a dark dot per phenomenon (mean across models, 95% bootstrap
     CI over assertions) plus a lighter colored dot per model.
  2. the same, broken down by group_id (each group is one phenomenon/subtype from pairs.csv),
     in separate critical and control panels (always both, regardless of --condition).
  3. the hardest assertions (lowest mean margin across models): a dark dot per assertion
     (mean margin across models, 95% bootstrap CI over models) plus a lighter colored dot
     per model. A margin above 0 means the model got the assertion right.
"""

import argparse
import glob
import os
import re
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

INK = "#0b0b0b"
MUTED = "#52514e"
# categorical slots 1-4 (blue, orange, aqua, yellow) from the dataviz reference palette
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#800080"]
MODEL_ALPHA = 0.7
N_BOOT = 10000
SEED = 0


def load_results(eval_dir):
    """Stack every eval_results-<model>.csv into one long DataFrame with a `model` column."""
    frames = []
    for path in sorted(glob.glob(os.path.join(eval_dir, "eval_results-*.csv"))):
        model = re.match(r"eval_results-(.+)\.csv", os.path.basename(path)).group(1)
        frames.append(pd.read_csv(path, keep_default_na=False, na_values=[]).assign(model=model))
    if not frames:
        raise SystemExit(f"No eval_results-*.csv files found in {eval_dir}")
    return pd.concat(frames, ignore_index=True)


def assertion_labels(pairs_path):
    """Map assertion_id -> 'unique_id / rank 1 diff > rank 2 diff' text from pairs.csv."""
    pairs = pd.read_csv(pairs_path, keep_default_na=False, na_values=[])
    labels = {}
    for assertion_id, group in pairs.groupby("assertion_id"):
        group = group.sort_values("rank")
        top, bottom = group.iloc[0], group.iloc[-1]
        lhs = f"{top['sentence1']} - {top['sentence2']}"
        rhs = f"{bottom['sentence1']} - {bottom['sentence2']}" if len(group) > 1 else "0"
        labels[assertion_id] = (
            f"{top['unique_id']} ({top['comparison_type']}, {top['condition_type']})\n"
            + lhs
            + "\n> "
            + rhs
        )
    return labels


def add_group_labels(results, pairs_path):
    """Attach a `group` column (group_id, phenomenon, subtype) from pairs.csv by assertion_id."""
    pairs = pd.read_csv(pairs_path, keep_default_na=False, na_values=[])
    info = pairs.drop_duplicates("assertion_id").set_index("assertion_id")

    def label(row):
        subtype = f"\n{textwrap.fill(row['subtype'], 16)}" if row["subtype"] else ""
        return f"{row['group_id']}\n{row['phenomenon']}{subtype}"

    return results.assign(group=results["assertion_id"].map(info.apply(label, axis=1)))


def style_axes(ax):
    sns.despine(ax=ax, left=True, bottom=True)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def add_legend(ax, models, palette, mean_label, **legend_kwargs):
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=9, color=INK, label=mean_label)
    ] + [
        Line2D([], [], marker="o", linestyle="none", markersize=7, color=palette[m],
               alpha=MODEL_ALPHA, label=m)
        for m in models
    ]
    ax.legend(handles=handles, frameon=False, labelcolor=MUTED, **legend_kwargs)


def draw_accuracy(ax, results, models, palette, by, order=None):
    """On ax: one dark mean dot + CI per `by` value, with a lighter dot per model."""
    per_assertion = results.groupby([by, "assertion_id"])["correct"].mean().reset_index()
    per_model = results.groupby([by, "model"])["correct"].mean().reset_index()
    counts = results.groupby(by)["assertion_id"].nunique()
    if order is None:
        order = per_assertion.groupby(by)["correct"].mean().sort_values(ascending=False).index

    ax.axhline(1.0, color=MUTED, lw=0.6, ls=":")
    sns.stripplot(
        data=per_model, x=by, y="correct", hue="model", order=order,
        hue_order=models, palette=palette, dodge=True, jitter=False, size=8, alpha=MODEL_ALPHA,
        edgecolor=INK, linewidth=0.4, legend=False, ax=ax,
    )
    sns.pointplot(
        data=per_assertion, x=by, y="correct", order=order, linestyle="none",
        color=INK, markersize=9, errorbar=("ci", 95), n_boot=N_BOOT, seed=SEED, capsize=0.12,
        err_kws={"linewidth": 1.6}, ax=ax,
    )
    ax.set_xticks(
        range(len(order)),
        [f"{k}\n(n={counts[k]})" if k in counts else f"{k}\n(none)" for k in order],
    )
    ax.set_xlabel("")
    ax.set_ylabel("accuracy")
    ax.set_ylim(top=1.03)
    ax.grid(axis="x", visible=False)
    style_axes(ax)


def plot_accuracy(results, models, palette, by, output, figsize, order=None):
    """Single-panel accuracy plot per `by` value."""
    fig, ax = plt.subplots(figsize=figsize)
    draw_accuracy(ax, results, models, palette, by, order)
    add_legend(ax, models, palette, "mean across models (95% CI)", loc="lower left")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_accuracy_by_condition(results, models, palette, by, output, figsize, order):
    """Stacked panels, critical on top and control below, sharing the same `by` order."""
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    for ax, condition, title in zip(
        axes, ["critical", "control"], ["critical assertions", "control assertions (sanity checks)"]
    ):
        draw_accuracy(ax, results[results["condition_type"] == condition], models, palette, by, order)
        ax.set_title(title, loc="left", color=INK)
    add_legend(axes[0], models, palette, "mean across models (95% CI)", loc="lower left")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_hardest_assertions(results, models, palette, labels, top_n, output):
    """The top_n assertions with the lowest mean margin: mean dot + CI, lighter dot per model."""
    all_hardest = results.groupby("assertion_id")["margin"].mean().sort_values()
    ranked = all_hardest.rename("mean_margin").reset_index()
    ranked.insert(1, "label", ranked["assertion_id"].map(labels).str.replace("\n", " "))
    ranked.to_csv(os.path.splitext(output)[0] + ".csv", index=False)
    hardest_n = all_hardest.head(top_n).index.tolist()
    data = results[results["assertion_id"].isin(hardest_n)].copy()
    data["label"] = data["assertion_id"].map(labels)
    order = [labels[a] for a in hardest_n]

    fig, ax = plt.subplots(figsize=(14, 0.95 * top_n + 1.5))
    ax.axvline(0, color=INK, lw=1)
    sns.stripplot(
        data=data, x="margin", y="label", hue="model", order=order, hue_order=models,
        palette=palette, dodge=True, jitter=False, size=8, alpha=MODEL_ALPHA,
        edgecolor=INK, linewidth=0.4, legend=False, orient="h", ax=ax,
    )
    sns.pointplot(
        data=data, x="margin", y="label", order=order, linestyle="none", color=INK,
        markersize=9, errorbar=("ci", 95), n_boot=N_BOOT, seed=SEED, capsize=0.12,
        err_kws={"linewidth": 1.6}, orient="h", ax=ax,
    )
    ax.set_xlim(right=max(data["margin"].max() + 0.1, 0.15))
    ax.set_xlabel("margin: rank 1 diff - rank 2 diff (log-prob per token); above 0 = correct")
    ax.set_ylabel("")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_title(f"{top_n} hardest assertions (lowest mean margin)", loc="left", color=INK, pad=34)
    style_axes(ax)
    add_legend(ax, models, palette, "mean across models (95% CI)", loc="lower right",
               bbox_to_anchor=(1.0, 1.0), ncol=len(models) + 1)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", default="eval_output")
    parser.add_argument("--pairs", default="benchmark_data/pairs.csv")
    parser.add_argument("--output-dir", default="eval_output")
    parser.add_argument(
        "--condition", choices=["critical", "control", "all"], default="critical",
        help="which assertions to plot (default: critical, matching eval.py's headline accuracy).",
    )
    parser.add_argument("--top-n", type=int, default=10, help="how many hardest assertions to plot.")
    args = parser.parse_args()

    all_results = add_group_labels(load_results(args.eval_dir), args.pairs)
    results = (
        all_results if args.condition == "all"
        else all_results[all_results["condition_type"] == args.condition]
    )
    models = sorted(results["model"].unique())
    palette = dict(zip(models, MODEL_COLORS))
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": MUTED, "grid.color": "#e4e3df"})

    plot_accuracy(
        results, models, palette, "phenomenon",
        os.path.join(args.output_dir, "plot_accuracy_by_phenomenon.png"), figsize=(9, 5.5),
    )
    group_order = all_results.sort_values("group")["group"].unique()
    plot_accuracy_by_condition(
        all_results, models, palette, "group",
        os.path.join(args.output_dir, "plot_accuracy_by_group.png"), figsize=(15, 11),
        order=group_order,
    )
    plot_hardest_assertions(
        results, models, palette, assertion_labels(args.pairs), args.top_n,
        os.path.join(args.output_dir, "plot_hardest_assertions.png"),
    )
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()

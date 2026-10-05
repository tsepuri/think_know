"""plot eval results (from eval.py) across every model found in eval_output/.

Three figures:
  1. accuracy by phenomenon: a dark dot per phenomenon (mean across models, 95% bootstrap
     CI over assertions) plus a lighter colored dot per model, with critical and control
     panels side by side (always both, regardless of --condition).
  2. the same, broken down by group_id (each group is one phenomenon/subtype from pairs.csv).
  3. the hardest assertions (lowest mean margin across models): a dark dot per assertion
     (mean margin across models, 95% bootstrap CI over models) plus a lighter colored dot
     per model. A margin above 0 means the model got the assertion right.

CHILDES-formatted results (eval.py --childes-format) are drawn as triangles next to each
model's circle; the dark mean dots and the hardest-assertion ranking use normative results only.
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

from utils import normative_to_childes_formatting

INK = "#0b0b0b"
MUTED = "#52514e"
# categorical slots 1-4 (blue, orange, aqua, yellow) from the dataviz reference palette
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#800080", "#28EF1A", "#FF0909"]
MODEL_ALPHA = 0.7
MODEL_DISPLAY_NAMES = {
    "BabyLM-community_BabyLM-2026-Baseline-GPT2-Strict": "babylm-100m-gpt2",
    "BabyLM-community_babylm-baseline-100m-gpt-bert-causal-focus": "babylm-100m-gpt-bert",
}
# eval.py --childes-format writes its outputs with this prefix; plotted as triangles
CHILDES_PREFIX = "childes_formatted_"
CHILDES_MARKER = "^"
N_BOOT = 10000
SEED = 0


def is_stale(eval_dir, prefix, model, pairs):
    """True if <prefix>pairs_scored-<model>.csv wasn't scored on the current pairs.csv
    (assertion ids get renumbered when pairs.csv is regenerated, so matching ids aren't enough)."""
    path = os.path.join(eval_dir, f"{prefix}pairs_scored-{model}.csv")
    if not os.path.exists(path):
        return False
    key = ["assertion_id", "rank", "sentence1", "sentence2"]
    scored = pd.read_csv(path, keep_default_na=False, na_values=[])[key]
    return not scored.sort_values(key).reset_index(drop=True).equals(
        pairs[key].sort_values(key).reset_index(drop=True)
    )


def load_results(eval_dir, pairs_path):
    """Stack every eval_results-<model>.csv (and childes_formatted_eval_results-<model>.csv)
    into one long DataFrame with `model` and `childes` columns, skipping results that were
    scored on an older pairs.csv."""
    pairs = pd.read_csv(pairs_path, keep_default_na=False, na_values=[])
    childes_pairs = pairs.assign(**{
        col: pairs[col].map(normative_to_childes_formatting) for col in ["sentence1", "sentence2"]
    })
    frames = []
    for prefix, expected, childes in [("", pairs, False), (CHILDES_PREFIX, childes_pairs, True)]:
        for path in sorted(glob.glob(os.path.join(eval_dir, f"{prefix}eval_results-*.csv"))):
            model = re.match(rf"{prefix}eval_results-(.+)\.csv", os.path.basename(path)).group(1)
            if is_stale(eval_dir, prefix, model, expected):
                print(f"warning: skipping {prefix}{model}: scored on a different {pairs_path} (re-run eval.py)")
                continue
            frames.append(pd.read_csv(path, keep_default_na=False, na_values=[]).assign(
                model=MODEL_DISPLAY_NAMES.get(model, model), childes=childes,
            ))
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


def add_legend(ax, models, palette, mean_label, childes=False, **legend_kwargs):
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=9, color=INK, label=mean_label)
    ] + [
        Line2D([], [], marker="o", linestyle="none", markersize=7, color=palette[m],
               alpha=MODEL_ALPHA, label=m)
        for m in models
    ]
    if childes:
        handles.append(Line2D([], [], marker=CHILDES_MARKER, linestyle="none", markersize=7,
                              color=MUTED, alpha=MODEL_ALPHA, label="CHILDES format"))
    ax.legend(handles=handles, frameon=False, labelcolor=MUTED, **legend_kwargs)


def draw_models(ax, data, models, palette, **kwargs):
    """One lighter dot per model: circles for normative results, triangles for CHILDES-formatted
    ones. Both share hue_order, so a model's triangle is dodged to the same spot as its circle."""
    for childes, marker in [(False, "o"), (True, CHILDES_MARKER)]:
        subset = data[data["childes"] == childes]
        if subset.empty:
            continue
        sns.stripplot(
            data=subset, hue="model", hue_order=models, palette=palette, marker=marker,
            dodge=True, jitter=False, size=8, alpha=MODEL_ALPHA, edgecolor=INK, linewidth=0.4,
            legend=False, ax=ax, **kwargs,
        )


def draw_accuracy(ax, results, models, palette, by, order=None):
    """On ax: one dark mean dot + CI per `by` value (normative results only), with a lighter
    dot per model."""
    normative = results[~results["childes"]]
    per_assertion = normative.groupby([by, "assertion_id"])["correct"].mean().reset_index()
    per_model = results.groupby([by, "model", "childes"])["correct"].mean().reset_index()
    counts = results.groupby(by)["assertion_id"].nunique()
    if order is None:
        order = per_assertion.groupby(by)["correct"].mean().sort_values(ascending=False).index

    ax.axhline(1.0, color=MUTED, lw=0.6, ls=":")
    draw_models(ax, per_model, models, palette, x=by, y="correct", order=order)
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
    # full 0-1 range, padded so dots at exactly 0 or 1 aren't clipped
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="x", visible=False)
    style_axes(ax)


def plot_accuracy_by_condition(results, models, palette, by, output, figsize, order):
    """Side-by-side panels, critical on the left and control on the right, sharing the y
    axis. The control panel only shows the `by` values that have control assertions, in
    the same order, and each panel's width is proportional to how many values it shows."""
    control = results[results["condition_type"] == "control"]
    control_order = [k for k in order if k in set(control[by])]
    fig, axes = plt.subplots(
        1, 2, figsize=figsize, sharey=True,
        gridspec_kw={"width_ratios": [len(order), max(len(control_order), 1)]},
    )
    for ax, condition, title, panel_order in zip(
        axes, ["critical", "control"], ["critical assertions", "control assertions"],
        [order, control_order],
    ):
        draw_accuracy(ax, results[results["condition_type"] == condition], models, palette, by, panel_order)
        ax.set_title(title, loc="left", color=INK)
    axes[1].set_ylabel("")
    add_legend(axes[1], models, palette, "mean across models (95% CI)",
               childes=results["childes"].any(), loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_hardest_assertions(results, models, palette, labels, top_n, output):
    """The top_n assertions with the lowest mean margin (normative results only): mean dot +
    CI, lighter dot per model."""
    normative = results[~results["childes"]]
    all_hardest = normative.groupby("assertion_id")["margin"].mean().sort_values()
    ranked = all_hardest.rename("mean_margin").reset_index()
    ranked.insert(1, "label", ranked["assertion_id"].map(labels).str.replace("\n", " "))
    ranked.to_csv(os.path.splitext(output)[0] + ".csv", index=False)
    hardest_n = all_hardest.head(top_n).index.tolist()
    data = results[results["assertion_id"].isin(hardest_n)].copy()
    data["label"] = data["assertion_id"].map(labels)
    order = [labels[a] for a in hardest_n]

    fig, ax = plt.subplots(figsize=(14, 0.95 * top_n + 1.5))
    ax.axvline(0, color=INK, lw=1)
    draw_models(ax, data, models, palette, x="margin", y="label", order=order, orient="h")
    sns.pointplot(
        data=data[~data["childes"]], x="margin", y="label", order=order, linestyle="none", color=INK,
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
    has_childes = data["childes"].any()
    add_legend(ax, models, palette, "mean across models (95% CI)", childes=has_childes,
               loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(models) + 1 + has_childes)
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
        help="which assertions the hardest-assertions plot uses (default: critical, matching "
             "eval.py's headline accuracy); the accuracy plots always show both.",
    )
    parser.add_argument("--top-n", type=int, default=10, help="how many hardest assertions to plot.")
    args = parser.parse_args()

    all_results = add_group_labels(load_results(args.eval_dir, args.pairs), args.pairs)
    results = (
        all_results if args.condition == "all"
        else all_results[all_results["condition_type"] == args.condition]
    )
    models = sorted(results["model"].unique())
    palette = dict(zip(models, MODEL_COLORS))
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": MUTED, "grid.color": "#e4e3df"})

    # order phenomena by mean critical accuracy, shared by the critical and control panels
    critical = all_results[(all_results["condition_type"] == "critical") & ~all_results["childes"]]
    phenomenon_order = (
        critical.groupby(["phenomenon", "assertion_id"])["correct"].mean()
        .groupby("phenomenon").mean().sort_values(ascending=False).index
    )
    plot_accuracy_by_condition(
        all_results, models, palette, "phenomenon",
        os.path.join(args.output_dir, "plot_accuracy_by_phenomenon.png"), figsize=(15, 5.5),
        order=phenomenon_order,
    )
    group_order = all_results.sort_values("group")["group"].unique()
    plot_accuracy_by_condition(
        all_results, models, palette, "group",
        os.path.join(args.output_dir, "plot_accuracy_by_group.png"), figsize=(24, 6.5),
        order=group_order,
    )
    plot_hardest_assertions(
        results, models, palette, assertion_labels(args.pairs), args.top_n,
        os.path.join(args.output_dir, "plot_hardest_assertions.png"),
    )
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()

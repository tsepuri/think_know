import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    ROOT = mo.notebook_dir().parent
    sys.path.insert(0, str(ROOT))
    import utils
    from scripts import plot_eval as pe
    from scripts import plot_pairwise_vs_differences as pvd

    return ROOT, mo, pd, pe, plt, pvd, sns, utils


@app.cell
def _(mo):
    mo.md(r"""
    # eval results across models

    Plots every model scored by `scripts/eval.py` (and `eval_slor_manual.py`) in `eval_output/`.
    CHILDES-formatted runs (`--childes-format`) are drawn as triangles beside each model's circle;
    dark mean dots and the hardest-assertion ranking use normative results only. Results scored on
    an older `pairs.csv` are skipped with a warning (re-run `eval.py`). Figures go to `figures/`.
    """)
    return


@app.cell
def _(mo):
    condition = mo.ui.dropdown(
        ["critical", "control", "all"], value="critical", label="hardest-assertions condition"
    )
    top_n = mo.ui.slider(5, 30, value=10, label="hardest assertions shown")
    mo.hstack([condition, top_n], justify="start")
    return condition, top_n


@app.cell
def _(ROOT, pe, sns):
    EVAL_DIR = ROOT / "eval_output"
    PAIRS = ROOT / "benchmark" / "data" / "pairs.csv"
    FIGURES = ROOT / "figures"
    all_results = pe.add_group_labels(pe.load_results(str(EVAL_DIR), str(PAIRS)), str(PAIRS))
    labels = pe.assertion_labels(str(PAIRS))
    models = sorted(all_results["model"].unique())
    palette = dict(zip(models, pe.MODEL_COLORS))
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": pe.MUTED, "grid.color": "#e4e3df"})
    return EVAL_DIR, FIGURES, all_results, labels, models, palette


@app.cell
def _(all_results):
    all_results.groupby(["model", "childes", "condition_type"])["correct"].mean().unstack()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Accuracy by phenomenon and group

    Critical assertions on the left, control on the right, sharing the y axis. Each dark dot is
    the mean across models with a 95% bootstrap CI over assertions; lighter dots are single models.
    Phenomena are ordered by mean critical accuracy; groups (`group_id` from `pairs.csv`) by id.
    """)
    return


@app.cell
def _(models, palette, pe, plt):
    def accuracy_by_condition(results, by, figsize, order):
        """pe.plot_accuracy_by_condition, returning the figure instead of saving it."""
        _control = results[results["condition_type"] == "control"]
        _control_order = [k for k in order if k in set(_control[by])]
        _fig, _axes = plt.subplots(
            1, 2, figsize=figsize, sharey=True,
            gridspec_kw={"width_ratios": [len(order), max(len(_control_order), 1)]},
        )
        for _ax, _condition, _panel_order in zip(_axes, ["critical", "control"], [order, _control_order]):
            pe.draw_accuracy(_ax, results[results["condition_type"] == _condition], models, palette, by, _panel_order)
            _ax.set_title(f"{_condition} assertions", loc="left", color=pe.INK)
        _axes[1].set_ylabel("")
        pe.add_legend(_axes[1], models, palette, "mean across models (95% CI)",
                      childes=results["childes"].any(), loc="upper left", bbox_to_anchor=(1.02, 1.0))
        _fig.tight_layout()
        return _fig

    return (accuracy_by_condition,)


@app.cell
def _(FIGURES, accuracy_by_condition, all_results, utils):
    # order phenomena by mean critical accuracy, shared by the critical and control panels
    _critical = all_results[(all_results["condition_type"] == "critical") & ~all_results["childes"]]
    _order = (
        _critical.groupby(["phenomenon", "assertion_id"])["correct"].mean()
        .groupby("phenomenon").mean().sort_values(ascending=False).index
    )
    _fig = accuracy_by_condition(all_results, "phenomenon", (15, 5.5), _order)
    utils.save_figure(_fig, str(FIGURES / "plot_accuracy_by_phenomenon.png"))
    return


@app.cell
def _(FIGURES, accuracy_by_condition, all_results, utils):
    _order = all_results.sort_values("group")["group"].unique()
    _fig = accuracy_by_condition(all_results, "group", (24, 6.5), _order)
    utils.save_figure(_fig, str(FIGURES / "plot_accuracy_by_group.png"))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Hardest assertions

    The assertions with the lowest mean margin across models (rank 1 diff $-$ rank 2 diff, in
    log-prob per token; above 0 means correct), restricted to the condition picked above. Dark
    dots carry a 95% bootstrap CI over models. The full ranking is written beside the figure as
    `plot_hardest_assertions.csv`.
    """)
    return


@app.cell
def _(
    FIGURES,
    all_results,
    condition,
    labels,
    models,
    palette,
    pe,
    plt,
    sns,
    top_n,
    utils,
):
    _results = all_results if condition.value == "all" else all_results[all_results["condition_type"] == condition.value]
    _ranking = _results[~_results["childes"]].groupby("assertion_id")["margin"].mean().sort_values()
    hardest_ranking = _ranking.rename("mean_margin").reset_index()
    hardest_ranking.insert(1, "label", hardest_ranking["assertion_id"].map(labels).str.replace("\n", " "))
    hardest_ranking.to_csv(FIGURES / "plot_hardest_assertions.csv", index=False)

    _hardest = _ranking.head(top_n.value).index.tolist()
    _data = _results[_results["assertion_id"].isin(_hardest)].assign(label=lambda d: d["assertion_id"].map(labels))
    _order = [labels[a] for a in _hardest]
    _fig, _ax = plt.subplots(figsize=(14, 0.95 * top_n.value + 1.5))
    _ax.axvline(0, color=pe.INK, lw=1)
    pe.draw_models(_ax, _data, models, palette, x="margin", y="label", order=_order, orient="h")
    sns.pointplot(
        data=_data[~_data["childes"]], x="margin", y="label", order=_order, linestyle="none", color=pe.INK,
        markersize=9, errorbar=("ci", 95), n_boot=pe.N_BOOT, seed=pe.SEED, capsize=0.12,
        err_kws={"linewidth": 1.6}, orient="h", ax=_ax,
    )
    _ax.set_xlim(right=max(_data["margin"].max() + 0.1, 0.15))
    _ax.set_xlabel("margin: rank 1 diff - rank 2 diff (log-prob per token); above 0 = correct")
    _ax.set_ylabel("")
    _ax.grid(axis="y", visible=False)
    _ax.tick_params(axis="y", labelsize=8)
    _ax.set_title(f"{top_n.value} hardest assertions (lowest mean margin)", loc="left", color=pe.INK, pad=34)
    pe.style_axes(_ax)
    _childes = _data["childes"].any()
    pe.add_legend(_ax, models, palette, "mean across models (95% CI)", childes=_childes,
                  loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=len(models) + 1 + _childes)
    _fig.tight_layout()
    utils.save_figure(_fig, str(FIGURES / "plot_hardest_assertions.png"))
    return (hardest_ranking,)


@app.cell
def _(hardest_ranking):
    hardest_ranking
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Differences vs. pairwise

    Accuracy on `pairs.csv` (differences of differences) against `pairs_pairwise.csv` (plain
    minimal pairs), one panel per model that has both `eval_results-<model>.csv` and
    `pairwise_eval_results-<model>.csv`: a row per phenomenon (critical), then overall critical
    and overall control, with the two accuracies joined by a line.
    """)
    return


@app.cell
def _(EVAL_DIR, FIGURES, pd, pe, plt, pvd, utils):
    _runs = pvd.load_runs(str(EVAL_DIR))
    _fig, _axes = plt.subplots(1, len(_runs), figsize=(3.2 * len(_runs) + 1.5, 4.2), sharey=True, sharex=True)
    _axes = [_axes] if len(_runs) == 1 else _axes
    for _ax, (_label, _differences, _pairwise) in zip(_axes, _runs):
        _table = pd.DataFrame({"differences": pvd.accuracy_rows(_differences), "pairwise": pvd.accuracy_rows(_pairwise)})
        _order = [r for r in _table.index if not r.startswith("Overall")] + ["Overall (critical)", "Overall (control)"]
        _table = _table.loc[_order[::-1]]
        _y = range(len(_table))
        _ax.hlines(_y, _table.min(axis=1), _table.max(axis=1), color=pe.MUTED, linewidth=1.5, alpha=0.5, zorder=1)
        for _version, _color in pvd.VERSION_COLORS.items():
            _ax.scatter(_table[_version], _y, s=50, color=_color, edgecolor="white", linewidth=1.5, zorder=2, label=_version)
        _ax.axhline(1.5, color=pe.MUTED, linewidth=0.8, alpha=0.4)
        _ax.set_yticks(list(_y), _table.index)
        _ax.set_title(_label, fontsize=10, color=pe.INK)
        _ax.set_xlim(0.5, 1.02)
        _ax.set_xticks([0.6, 0.8, 1.0])
        _ax.set_xlabel("accuracy", color=pe.MUTED)
        pe.style_axes(_ax)
    _axes[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _fig.suptitle("Accuracy: differences of differences vs pairwise minimal pairs", color=pe.INK)
    _fig.tight_layout()
    utils.save_figure(_fig, str(FIGURES / "plot_pairwise_vs_differences.png"))
    return


if __name__ == "__main__":
    app.run()

"""
Publication figures, drawn from stored results.

Six figures, each answering one question the paper asks. Diagnostic plots the
pipeline already emits -- calibration per model, residual distributions -- stay
where they are as supplementary material; a figure earns a place here only by
carrying part of the argument.

Every figure ships with direct labels and a companion table. That is not
redundancy: one validated palette slot sits below the 3:1 contrast gate, and
the relief rule requires labels or a table view. Both are provided.
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.logging import get_logger
from src.reporting.style import (
    AXIS,
    INK_MUTED,
    INK_SECONDARY,
    REFERENCE_LINE,
    SEQUENTIAL,
    annotate_bars,
    save,
    series_colour,
)
from src.reporting.tables import MODEL_FAMILY, MODEL_ORDER

logger = get_logger(__name__)


def figure_protocol_collapse(protocol: pd.DataFrame, output: Path) -> None:
    """
    The central result: identical models, protocol the only difference.

    Grouped bars rather than a line, because the three protocols are
    categories, not an ordered scale -- a line would imply an interpolation
    between them that does not exist.
    """
    models = protocol["Model"].tolist()
    positions = np.arange(len(models))
    width = 0.24

    fig, ax = plt.subplots(figsize=(7.2, 3.9))

    for index, (column, label) in enumerate([
        ("Retrospective R2", "Retrospective"),
        ("Forward R2", "Forward"),
        ("Walk-forward R2", "Walk-forward"),
    ]):
        values = protocol[column].to_numpy()
        bars = ax.bar(
            positions + (index - 1) * width, values, width * 0.88,
            label=label, color=series_colour(index), linewidth=0,
        )
        annotate_bars(ax, bars, values, fmt="{:.2f}")

    ax.set_xticks(positions)
    ax.set_xticklabels(models)
    ax.set_ylabel("R² (out-of-sample)")
    # Headroom above the tallest bar so its label cannot meet the legend.
    ax.set_ylim(0, 1.12)

    # Title, explanation and legend each get their own horizontal band above
    # the axes. Stacking them by eye put the legend through the subtitle, and
    # an in-axes legend sat across the bar labels it was meant to explain.
    fig.subplots_adjust(top=0.76)
    fig.suptitle(
        "Apparent performance under three evaluation protocols",
        y=0.985, fontsize=10.5,
    )
    fig.text(
        0.5, 0.905,
        "Same models, same features, untuned in every arm — only the protocol differs",
        ha="center", fontsize=8, color=INK_MUTED,
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=3,
              borderaxespad=0.0)

    save(fig, output)


def figure_metric_disagreement(summary: pd.DataFrame, output: Path) -> None:
    """
    Which metric is reported changes the ranking.

    A slope chart: rank by MAE on the left, rank by RMSE on the right. Crossing
    lines are the finding -- the orderings are not merely different in spacing
    but inverted between model families.
    """
    frame = summary[summary["model"].isin(MODEL_ORDER)].copy()
    frame = frame[frame["model"] != "Drift"]  # an outlier that compresses the rest

    frame["mae_rank"] = frame["MAE"].rank()
    frame["rmse_rank"] = frame["RMSE"].rank()

    family_colour = {
        "Temporal baseline": series_colour(0),
        "Traditional equation": series_colour(1),
        "Learned": series_colour(2),
    }

    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    for _, row in frame.iterrows():
        family = MODEL_FAMILY.get(row["model"], "Learned")
        ax.plot(
            [0, 1], [row["mae_rank"], row["rmse_rank"]],
            color=family_colour[family], marker="o", markersize=4,
        )
        ax.annotate(
            f"{row['model']}  {row['MAE']:.1f}",
            xy=(-0.03, row["mae_rank"]), ha="right", va="center",
            fontsize=7.5, color=INK_SECONDARY,
        )
        ax.annotate(
            f"{row['RMSE']:.1f}  {row['model']}",
            xy=(1.03, row["rmse_rank"]), ha="left", va="center",
            fontsize=7.5, color=INK_SECONDARY,
        )

    handles = [
        plt.Line2D([], [], color=colour, marker="o", markersize=4, label=family)
        for family, colour in family_colour.items()
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.16), ncols=3)

    ax.set_xlim(-0.55, 1.55)
    ax.invert_yaxis()
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ranked by MAE", "ranked by RMSE"])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.set_title("The ranking depends on which error is reported")

    save(fig, output)


def figure_ablation(ablation_summary: pd.DataFrame, baseline_mae: float,
                    output: Path) -> None:
    """
    Information against architecture.

    Bars darken with feature count -- an ordered sequence, so a sequential ramp
    rather than categorical hues. The baseline reference line is what makes the
    figure an argument instead of a list: it shows where simple continuation
    sits relative to every configuration.
    """
    frame = ablation_summary.copy()
    positions = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(6.8, 3.8))

    colours = [SEQUENTIAL[min(i, len(SEQUENTIAL) - 1)] for i in range(len(frame))]
    bars = ax.bar(positions, frame["best_MAE"], 0.62, color=colours, linewidth=0)
    annotate_bars(ax, bars, frame["best_MAE"].to_numpy())

    # The reference line is named in the caption band rather than labelled in
    # place. Every configuration past the first lands within a few kg of it, so
    # an in-plot label necessarily overlaps a bar whichever side it is placed.
    ax.axhline(
        baseline_mae, color=REFERENCE_LINE, linestyle="--", linewidth=1.2,
        label=f"best temporal baseline ({baseline_mae:.1f} kg)",
    )
    # Upper right: the only region clear of both bars and the reference line.
    ax.legend(loc="upper right", fontsize=7.5)

    labels = [
        f"{name.replace('_', ' ')}\n({n} features)"
        for name, n in zip(frame["feature_set"], frame["n_features"])
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE of best learned model (kg)")
    ax.set_ylim(0, frame["best_MAE"].max() * 1.12)

    # The summary numbers go above the axes, not inside them: placed in the
    # plot area they ran across the tallest bar.
    gains = frame["gain_over_demographics"].to_numpy()
    spread = frame["spread_across_models"].to_numpy()

    fig.subplots_adjust(top=0.80)
    fig.suptitle(
        "What information buys, against what algorithm choice buys",
        y=0.985, fontsize=10.5,
    )
    fig.text(
        0.5, 0.885,
        f"Previous total alone: {gains[1]:.1f} kg    "
        f"every further feature combined: {gains[-1] - gains[1]:.1f} kg    "
        f"switching algorithm: {spread.min():.1f}–{spread.max():.1f} kg",
        ha="center", fontsize=8, color=INK_MUTED,
    )

    save(fig, output)


def figure_coverage_effect(full: pd.DataFrame, matched: pd.DataFrame,
                           output: Path) -> None:
    """
    Why the traditional comparison must be made on shared data.

    Paired bars per model: pooled over everything, against the subpopulation
    every model could score. The traditional equations do not move, because
    that subpopulation is all they ever addressed; the others improve, which is
    what reveals the subpopulation as easier rather than the equations as
    better.
    """
    full_mae = full.set_index("model")["MAE"]
    matched_mae = matched.set_index("model")["MAE"]

    models = [m for m in MODEL_ORDER if m in matched_mae.index and m != "Drift"]
    positions = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.0, 4.4))

    full_values = [full_mae[m] for m in models]
    matched_values = [matched_mae[m] for m in models]

    bars_full = ax.bar(positions - width / 2, full_values, width * 0.90,
                       label="All predictions", color=series_colour(0), linewidth=0)
    bars_matched = ax.bar(positions + width / 2, matched_values, width * 0.90,
                          label="Shared observations only", color=series_colour(1),
                          linewidth=0)

    annotate_bars(ax, bars_full, full_values)
    annotate_bars(ax, bars_matched, matched_values)

    ax.set_xticks(positions)
    # Steeper rotation and right alignment: at 20 degrees the longer model
    # names overlapped their neighbours.
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.set_ylabel("MAE (kg)")
    ax.set_ylim(0, max(max(full_values), max(matched_values)) * 1.16)

    fig.subplots_adjust(top=0.78, bottom=0.22)
    fig.suptitle(
        "Restricting every model to the observations all of them scored",
        y=0.985, fontsize=10.5,
    )
    fig.text(
        0.5, 0.90,
        "The traditional equations do not move — that subpopulation is all they ever scored",
        ha="center", fontsize=8, color=INK_MUTED,
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=2,
              borderaxespad=0.0)

    save(fig, output)


def figure_temporal_stability(temporal: pd.DataFrame, models: list,
                              output: Path) -> None:
    """
    Error by forecast year, with the weight behind each point.

    Two stacked panels sharing one x-axis -- never a second y-scale on the same
    panel. The lower panel is what stops the upper one being misread: the early
    windows carry almost no predictions, so their swings are noise.
    """
    frame = temporal[temporal["model"].isin(models)]

    fig, (ax_error, ax_count) = plt.subplots(
        2, 1, figsize=(7.2, 4.6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )

    for index, model in enumerate(models):
        series = frame[frame["model"] == model].sort_values("forecast_window")
        ax_error.plot(
            series["forecast_window"], series["MAE"],
            label=model, color=series_colour(index), marker="o", markersize=3,
        )

    ax_error.set_ylabel("MAE (kg)")
    # Legend above the panel: placed inside it, the early-period error spikes
    # reach the top of the axes and run straight through it.
    ax_error.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
                    ncols=len(models), borderaxespad=0.0)

    fig.subplots_adjust(top=0.80)
    fig.suptitle(
        "Forecast error by year, and the predictions behind it",
        y=0.985, fontsize=10.5,
    )
    fig.text(
        0.5, 0.905,
        "Early windows swing because almost nothing is being predicted — "
        "read the upper panel against the lower",
        ha="center", fontsize=8, color=INK_MUTED,
    )

    counts = (
        frame[frame["model"] == models[0]].sort_values("forecast_window")
    )
    ax_count.fill_between(
        counts["forecast_window"], counts["prediction_count"],
        color=INK_MUTED, alpha=0.35, linewidth=0,
    )
    ax_count.set_ylabel("n")
    ax_count.set_xlabel("Forecast window")
    ax_count.grid(False)

    save(fig, output)


def figure_subgroup_disparity(disparities: pd.DataFrame, model: str,
                              output: Path) -> None:
    """
    Spread between best- and worst-served subgroup, per dimension.

    One model, so the comparison is across dimensions rather than confounded by
    model choice. Sample sizes are printed against each bar: a gap between
    groups of very different size is frequently about size.
    """
    frame = disparities[disparities["model"] == model].copy()
    if frame.empty:
        logger.warning("No disparity rows for %s; skipping figure.", model)
        return

    frame = frame.sort_values("MAE_gap")
    positions = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(6.8, 0.52 * len(frame) + 1.8))

    bars = ax.barh(positions, frame["MAE_gap"], 0.6,
                   color=series_colour(0), linewidth=0)

    for bar, (_, row) in zip(bars, frame.iterrows()):
        # Named rather than arrowed: "best ... worst ..." says which end is
        # which without relying on a glyph, and the project holds its source to
        # the Windows console codepage so that log output cannot break.
        ax.annotate(
            f"{row['MAE_gap']:.1f} kg   "
            f"best {row['best_subgroup']} ({row['best_n']:,})   "
            f"worst {row['worst_subgroup']} ({row['worst_n']:,})",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(4, 0), textcoords="offset points",
            va="center", ha="left", fontsize=7.5, color=INK_SECONDARY,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels([d.replace("_", " ") for d in frame["dimension"]])
    ax.set_xlabel("MAE gap between best- and worst-served subgroup (kg)")
    ax.set_title(f"Subgroup disparity — {model}")
    ax.grid(False)
    ax.set_xlim(0, frame["MAE_gap"].max() * 2.15)

    # Explicit left margin: dimension names are long, and without a tight
    # bounding box the default margin clips them.
    fig.subplots_adjust(left=0.22, right=0.97)

    save(fig, output)

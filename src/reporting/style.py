"""
Shared visual style for publication figures.

Defined once so every figure in the paper is consistent by construction rather
than by remembering to match the last one. Figures are static and print-bound,
so there is no interaction layer and no dark mode; the colour method, mark
specifications and accessibility rules still apply.

Palette
-------
Categorical slots are assigned in fixed order and never cycled. The three
leading slots were validated as a set against the light chart surface -- all
pairs, both colour-vision-deficiency separation and normal-vision separation --
rather than chosen by eye:

    worst all-pairs CVD separation       dE 9.2  (target >= 8)
    worst all-pairs normal-vision        dE 24.0 (floor >= 15)

Aqua measures 2.74:1 against the surface, below the 3:1 contrast gate. The
relief rule therefore applies and is satisfied twice over: every figure carries
visible direct labels, and every figure ships beside a table of the same
numbers.

Beyond three categories, these gates cannot be met by any ordering of the full
palette. Figures needing more use small multiples or fold the remainder
together, rather than reaching for a fourth hue.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots, fixed order. Never cycle; never reorder per chart.
SERIES = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
}
SERIES_ORDER = ["blue", "orange", "aqua"]

# Sequential ramp for ordered magnitude (feature sets, deciles). One hue,
# light to dark. For ordinal marks the lightest step must still clear 2:1
# against the surface, so this starts at step 250 rather than 100.
SEQUENTIAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

# Chrome and ink. Text never wears a series colour.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# Reserved for the reference line that marks "what simple continuation
# achieves" -- a threshold, not a series, so it is deliberately neutral.
REFERENCE_LINE = "#52514e"


def series_colour(index: int) -> str:
    """Categorical colour by position. Raises past the validated set."""
    if index >= len(SERIES_ORDER):
        raise ValueError(
            f"Requested categorical slot {index}; only {len(SERIES_ORDER)} are "
            "validated for all-pairs separation. Use small multiples or fold "
            "the remaining categories together rather than adding a hue."
        )
    return SERIES[SERIES_ORDER[index]]


def apply_style() -> None:
    """Install the publication defaults on matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": None,
        "savefig.dpi": 300,

        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,

        "text.color": INK_PRIMARY,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,

        # Recessive chrome: the data carries the figure, not the frame.
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,

        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "legend.frameon": False,
        "figure.autolayout": False,
    })


def save(fig, path: Path) -> None:
    """
    Write a figure as both raster and vector.

    PDF is what a typesetter should receive; PNG is for reading drafts and for
    anything that cannot place vector art.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def annotate_bars(ax, bars, values, fmt="{:.1f}", offset=2.0) -> None:
    """
    Label bars directly.

    Not decoration: the validated palette puts one slot below the 3:1 contrast
    gate, and direct labelling is the relief that makes identity readable
    without relying on colour.
    """
    for bar, value in zip(bars, values):
        ax.annotate(
            fmt.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK_SECONDARY,
        )

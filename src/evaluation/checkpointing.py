"""
Incremental persistence for long walk-forward runs.

The evaluator accumulates every prediction in memory and writes only after the
final fold. A production run takes roughly 50 minutes for the raw cohort and
two hours for equipped, so a crash, an out-of-memory kill, or a machine sleeping
during the last fold discards all of it. That matters more for ablation, which
is several walk-forward runs in sequence.

Completed folds are written to disk as they finish. A resumed run reads them
back and skips those folds, bounding the cost of an interruption to one fold.

Format
------
One binary frame per fold, rather than appended rows in a shared CSV. A resumed
run must produce results indistinguishable from an uninterrupted one, and CSV
does not round-trip a DataFrame exactly: it widens int32 to int64, turns
timestamps into strings, and loses the last bits of a float. Measured against
an uninterrupted run, that drift reached 1e-13 in predicted totals -- small,
but this project exists to make results reproducible, so a checkpoint that
quietly perturbs them is not acceptable.

Pickle is used because it is exactly lossless and needs nothing beyond pandas.
These files are transient local working artefacts: written and read by this
same code, never published, and deleted once the authoritative outputs land.
They should not be loaded from an untrusted source.
"""
import json
import re
from pathlib import Path
from typing import List, Set

import pandas as pd

from src.core.logging import get_logger

logger = get_logger(__name__)

CHECKPOINT_DIRNAME = "walk_forward_checkpoint"
CHECKPOINT_STATE_FILENAME = "walk_forward_checkpoint_state.json"


def _as_native(value):
    """
    Convert a numpy scalar to its Python equivalent for JSON serialisation.

    Forecast windows come from `Date.dt.year`, which yields numpy int32. json
    cannot encode those, so writing the state file raised mid-run -- after the
    fold's predictions had already been written, leaving the checkpoint
    inconsistent with its own state.
    """
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def _window_filename(window) -> str:
    """Filesystem-safe name for a fold, keeping the window visible in the name."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(_as_native(window)))
    return f"window_{safe}.pkl"


class WalkForwardCheckpoint:
    """
    Record of completed folds, one file each.

    Identified by the configuration that produced it. Resuming into a
    checkpoint written under a different feature or model set would splice
    incompatible runs together, so the fingerprint is verified before any fold
    is skipped.
    """

    def __init__(self, directory: Path, fingerprint: str, enabled: bool = True):
        self.directory = Path(directory)
        self.fingerprint = fingerprint
        self.enabled = enabled
        self.frames_dir = self.directory / CHECKPOINT_DIRNAME
        self.state_path = self.directory / CHECKPOINT_STATE_FILENAME

    # -- reading ---------------------------------------------------------

    def completed_windows(self) -> Set:
        """Forecast windows already written under this exact configuration."""
        if not self.enabled or not self.state_path.exists():
            return set()

        state = json.loads(self.state_path.read_text(encoding="utf-8"))

        if state.get("fingerprint") != self.fingerprint:
            logger.warning(
                "Checkpoint at %s was written under a different configuration; "
                "ignoring it and starting fresh.",
                self.directory,
            )
            return set()

        return set(state.get("completed_windows", []))

    def load_predictions(self) -> List[dict]:
        """Every checkpointed prediction, with original dtypes intact."""
        if not self.enabled:
            return []

        windows = self.completed_windows()
        if not windows:
            return []

        frames = []
        for window in sorted(windows, key=str):
            path = self.frames_dir / _window_filename(window)
            if not path.exists():
                logger.warning(
                    "Window %s is marked complete but its data is missing; "
                    "it will be recomputed.", window,
                )
                continue
            frames.append(pd.read_pickle(path))

        if not frames:
            return []

        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            "Resuming from checkpoint: %d predictions across %d completed folds.",
            len(combined), len(frames),
        )
        return combined.to_dict("records")

    def recovered_windows(self) -> Set:
        """Windows whose data is genuinely present, not merely marked complete."""
        if not self.enabled:
            return set()
        return {
            window for window in self.completed_windows()
            if (self.frames_dir / _window_filename(window)).exists()
        }

    # -- writing ---------------------------------------------------------

    def record_window(self, window, predictions: List[dict]) -> None:
        """Persist one completed fold, then mark it complete."""
        if not self.enabled or not predictions:
            return

        self.frames_dir.mkdir(parents=True, exist_ok=True)

        frame = pd.DataFrame(predictions)
        frame.to_pickle(self.frames_dir / _window_filename(window))

        # State is written after the data, never before: a crash between the
        # two leaves the fold unmarked and simply recomputed, whereas the
        # reverse order would skip a fold whose data never landed.
        completed = self.completed_windows()
        completed.add(_as_native(window))
        self.state_path.write_text(
            json.dumps(
                {
                    "fingerprint": self.fingerprint,
                    "completed_windows": sorted(completed, key=str),
                },
                indent=4,
            ),
            encoding="utf-8",
        )

    def clear(self) -> None:
        """Remove the checkpoint once authoritative outputs have been written."""
        if self.frames_dir.exists():
            for path in self.frames_dir.glob("window_*.pkl"):
                path.unlink()
            try:
                self.frames_dir.rmdir()
            except OSError:
                # Something unexpected is in there; leave it rather than guess.
                logger.warning(
                    "Checkpoint directory %s is not empty; left in place.",
                    self.frames_dir,
                )

        if self.state_path.exists():
            self.state_path.unlink()

        logger.info("Cleared walk-forward checkpoint in %s", self.directory)


def build_fingerprint(feature_cols: List[str], model_names: List[str],
                      granularity: str, min_train_periods: int) -> str:
    """
    Identify the configuration a checkpoint belongs to.

    Everything that changes what a fold computes is included. Resuming across a
    change to any of these would combine folds computed under different
    experimental conditions into one result set.
    """
    parts = [
        "features=" + ",".join(sorted(feature_cols)),
        "models=" + ",".join(sorted(model_names)),
        f"granularity={granularity}",
        f"min_train_periods={min_train_periods}",
    ]
    return "|".join(parts)

import matplotlib

# Force the non-interactive backend before any test imports pyplot.
# Several modules under test save figures; without this, matplotlib selects an
# interactive backend and fails in headless environments (CI, and any local run
# without a display) with _tkinter.TclError.
matplotlib.use("Agg")

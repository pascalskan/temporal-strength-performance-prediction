"""
Project package initialisation.

Selects a non-interactive matplotlib backend before any submodule imports
pyplot. Every figure this project produces is written to disk, never shown, so
an interactive backend has nothing to offer and actively causes failures: it
opens Tk windows during pipeline runs (raising "main thread is not in main
loop" on teardown) and aborts outright in headless environments such as CI.
"""
import matplotlib

matplotlib.use("Agg")

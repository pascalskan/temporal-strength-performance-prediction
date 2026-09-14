"""
Source must stay encodable in the Windows console's default codepage.

Log messages containing emoji and typographic characters raised
UnicodeEncodeError under cp1252 on every run. Python's logging swallows the
exception, so the pipeline still completed, but each affected call printed a
"--- Logging error ---" traceback instead of its message, including the
walk-forward "all outputs validated" line. A clean run looked like a failing
one, and the actual message was lost.
"""
import pathlib
import unicodedata

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]

SOURCE_FILES = sorted(
    list((PROJECT_ROOT / "src").rglob("*.py"))
    + list(PROJECT_ROOT.glob("*.py"))
)


def _unencodable(text):
    offenders = []
    for char in text:
        try:
            char.encode("cp1252")
        except UnicodeEncodeError:
            offenders.append(char)
    return offenders


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: p.name)
def test_source_is_cp1252_encodable(path):
    offenders = _unencodable(path.read_text(encoding="utf-8"))

    if offenders:
        described = ", ".join(
            f"U+{ord(c):04X} ({unicodedata.name(c, 'unnamed')})"
            for c in dict.fromkeys(offenders)
        )
        pytest.fail(
            f"{path.relative_to(PROJECT_ROOT)} contains characters that cannot "
            f"encode to cp1252: {described}. Use ASCII equivalents in log "
            f"messages ('->' rather than an arrow, 'Delta' rather than the "
            f"Greek letter)."
        )


def test_matplotlib_uses_a_non_interactive_backend():
    """
    Figures are always written to disk, never displayed. An interactive backend
    opens Tk windows mid-run and fails outright in headless environments such
    as CI.
    """
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"

"""
Unit tests for build_manifest.grade_group().

Run with:
    python -m pytest tests/test_grade_group.py -v

No data files are required.
"""

import sys
from pathlib import Path

# Ensure src/ is on the path regardless of where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_manifest import grade_group


# ── ISUP Grade Group mapping ───────────────────────────────────────────────────

def test_gg1_3_3():
    assert grade_group("3+3") == 1

def test_gg2_3_4():
    assert grade_group("3+4") == 2

def test_gg3_4_3():
    assert grade_group("4+3") == 3

def test_gg4_4_4():
    assert grade_group("4+4") == 4

def test_gg4_3_5():
    # Previously misclassified as GG2 in the pipeline reconstruction
    assert grade_group("3+5") == 4

def test_gg4_5_3():
    assert grade_group("5+3") == 4

def test_gg5_4_5():
    assert grade_group("4+5") == 5

def test_gg5_5_4():
    assert grade_group("5+4") == 5

def test_gg5_5_5():
    assert grade_group("5+5") == 5


# ── Robustness: spaces around the '+' ─────────────────────────────────────────

def test_spaces_3_4():
    assert grade_group("3 + 4") == 2

def test_spaces_3_5():
    assert grade_group("3 + 5") == 4

def test_spaces_4_3():
    assert grade_group("4 + 3") == 3

def test_leading_trailing_spaces():
    assert grade_group("  3+4  ") == 2


# ── Edge cases: missing / benign ───────────────────────────────────────────────

def test_no_plus_returns_0():
    assert grade_group("nan") == 0

def test_empty_string_returns_0():
    assert grade_group("") == 0

def test_nan_string_returns_0():
    assert grade_group("nan") == 0

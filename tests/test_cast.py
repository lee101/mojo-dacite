"""Parity tests for the numeric half of dacite's ``Config.cast``.

dacite's cast is ``data = type_(data)``. For the numeric targets that is
arithmetic, and it has two behaviours a plausible kernel gets wrong:

* ``int(v)`` truncates **toward zero**, not down, so ``int(-1.5)`` is ``-1``
  and not ``-2``;
* a Python ``int`` is unbounded, so ``int(2**70)`` succeeds where a kernel
  storing into int64 would wrap. The kernel flags those instead, and the shim
  recomputes them with Python's ``int``.

The reference is Python's own conversion, which is what dacite's ``type_(data)``
evaluates to.
"""

import math

import pytest

from mojo_dacite import cast_numeric

FLOATS = [
    0.0, -0.0, 1.0, -1.0, 1.5, -1.5, 2.5, -2.5, 0.5, -0.5, 3.9999, -3.9999,
    1e18, -1e18, 1e300, -1e300, 1e-300, 123456.789, -123456.789,
    2.0**62, 2.0**63, 2.0**63 + 1.0, -(2.0**63), -(2.0**63) - 1.0,
]
BIG_INTS = [0, 1, -1, 2**53, 2**53 + 1, -(2**53), 2**62, 2**63, 2**63 + 1,
            2**64, 2**70, -(2**70), 10**30, -(10**30)]


@pytest.mark.parametrize("target", [int, float, bool])
def test_cast_float_matches_python(target):
    got = cast_numeric(FLOATS, target)
    assert got == [target(v) for v in FLOATS]


@pytest.mark.parametrize("target", [int, float, bool])
def test_cast_int_matches_python(target):
    """An int64-representable int is exact in float64, so this run is exact.
    An int beyond 2**53 is not, and the kernel declines to guess."""
    representable = [v for v in BIG_INTS if -(2**53) <= v <= 2**53]
    got = cast_numeric(representable, target)
    assert got == [target(v) for v in representable]


def test_int_beyond_float64_is_still_exact_after_cast():
    """int(2**53 + 1) is 9007199254740993, but Float64(2**53 + 1) is
    9007199254740992. dacite casting an int to int is a no-op, so the port
    must not route it through float64 and come back with a different number."""
    values = [2**53 + 1, -(2**53) - 1, 2**62, 10**30]
    assert cast_numeric(values, int) == values
    assert cast_numeric(values, float) == [float(v) for v in values]


def test_int_outside_int64_is_delegated():
    """The kernel's result is an int64, so a value outside that range is
    flagged and recomputed by Python. Silently wrapping it would turn
    2**64 into -9223372036854775808."""
    for v in (2**63, 2**64, 2**70, -(2**64), -(2**70)):
        assert cast_numeric([v], int) == [v]
    got = cast_numeric([1.0, 2.0**63, 3.0], int)
    assert got == [1, 2**63, 3]


def test_truncation_is_toward_zero():
    """The single most likely kernel bug in this file: floor instead of
    trunc, which disagrees on every negative non-integral value."""
    values = [-1.5, -2.5, -0.5, -0.9, -1.9, -123.456]
    assert cast_numeric(values, int) == [int(v) for v in values]
    assert cast_numeric(values, int) == [-1, -2, 0, 0, -1, -123]


def test_bool_collapse():
    values = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, float("nan"), 1e300]
    assert cast_numeric(values, bool) == [bool(v) for v in values]
    assert cast_numeric([0.0, -0.0], bool) == [False, False]
    assert cast_numeric([float("nan")], bool) == [True]


def test_nan_and_infinity_raise_for_int_like_python():
    """Python has no int for these, and dacite's int(nan) raises, so the port
    must raise the same way rather than returning a flagged zero."""
    for v in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises((ValueError, OverflowError)):
            cast_numeric([v], int)
    # float and bool accept them.
    assert cast_numeric([float("inf")], float) == [float("inf")]
    assert cast_numeric([float("nan")], bool) == [True]


def test_complex_is_passed_through_to_python():
    values = [1, 1.5, 2 + 3j, 1e300]
    assert cast_numeric(values, complex) == [complex(v) for v in values]


def test_empty_run():
    for target in (int, float, bool, complex):
        assert cast_numeric([], target) == []


def test_large_run_matches_python():
    import numpy as np

    rng = np.random.default_rng(11)
    values = (rng.standard_normal(100000) * 1e6).tolist()
    assert cast_numeric(values, int) == [int(v) for v in values]
    assert cast_numeric(values, float) == values

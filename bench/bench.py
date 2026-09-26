"""Correctness-gated benchmark for mojo-dacite.

Every case checks agreement with ``dacite.types.is_instance`` or with Python's
own conversion before timing. The baselines are the two fastest fair
alternatives: a vectorised NumPy comparison for the tower (the same rank
comparison, done with array ops) and a NumPy vectorised truncation for the
cast.

The scalar case is on the list too, and it is a loss: dacite's own
``is_instance`` is a handful of C-level ``isinstance`` calls, and no FFI
boundary can be cheaper than that. It is reported because omitting it would
flatter the port.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_dacite as mdac  # noqa: E402

# Tower ranks, as the kernel sees them.
RANK = {bool: 1, int: 2, float: 3, complex: 4}


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_column(n: int = 1 << 22):
    """A whole column of values against one annotated numeric field, which is
    the shape the kernel exists for."""
    rng = np.random.default_rng(0)
    values = rng.standard_normal(n).tolist()
    declared = float
    rank = np.full(n, RANK[declared], dtype=np.int32)

    kinds = np.full(n, RANK[declared], dtype=np.int32)
    got = mdac.column_accept_mask(declared, values, kinds=kinds)
    assert got.all(), "every standard normal is a float"

    from dacite.types import is_instance

    for i in range(0, n, n // 200):
        assert bool(got[i]) == is_instance(values[i], declared), i

    def numpy_side():
        return np.fromiter(
            (RANK[type(v)] <= RANK[declared] for v in values),
            dtype=bool, count=n,
        )

    def mojo_side():
        return mdac.column_accept_mask(declared, values, kinds=kinds)

    # Correctness gate: the two agree element for element.
    np.testing.assert_array_equal(mojo_side(), numpy_side())
    return f"column_accept_mask n={n}", _time(numpy_side, 3), _time(mojo_side, 3)


def bench_column_rejects(n: int = 1 << 22):
    """The same column, but with a mix of accepted and rejected values so the
    kernel's rejection bookkeeping is exercised rather than a run of Trues."""
    rng = np.random.default_rng(1)
    kinds = rng.integers(1, 5, size=n)
    rank = kinds.astype(np.int32)
    declared = float
    out = np.empty(n, dtype=np.uint8)

    from mojo_dacite import _lib

    _lib.lib.dac_tower_check_one(
        RANK[declared], rank.ctypes.data, n, out.ctypes.data
    )
    expect = rank <= RANK[declared]
    assert np.array_equal(out.astype(bool), expect)
    rejected = int(expect.size - expect.sum())
    assert rejected > n // 8, rejected

    def numpy_side():
        return rank <= RANK[declared]

    def mojo_side():
        _lib.lib.dac_tower_check_one(
            RANK[declared], rank.ctypes.data, n, out.ctypes.data
        )
        return out.astype(bool)

    mojo_side()
    np.testing.assert_array_equal(mojo_side(), numpy_side())
    return f"tower check mixed n={n}", _time(numpy_side, 3), _time(mojo_side, 3)


def bench_cast_int(n: int = 1 << 22):
    """The cast to int, which is truncation toward zero, against the vectorised
    NumPy truncation."""
    rng = np.random.default_rng(2)
    arr = rng.standard_normal(n) * 1e6
    values = arr.tolist()

    got = mdac.cast_numeric_array(arr, int)
    expect = [int(v) for v in values]
    assert got.tolist() == expect, "cast must match Python exactly"

    def numpy_side():
        return np.trunc(arr).astype(np.int64)

    def mojo_side():
        return mdac.cast_numeric_array(arr, int)

    return f"cast to int n={n}", _time(numpy_side, 3), _time(mojo_side, 3)


def bench_cast_bool(n: int = 1 << 22):
    a = np.random.default_rng(3).standard_normal(n) * 1000
    got = mdac.cast_numeric_array(a, bool)
    assert got.tolist() == [bool(v) for v in a.tolist()]

    def numpy_side():
        return a != 0.0

    def mojo_side():
        return mdac.cast_numeric_array(a, bool)

    return f"cast to bool n={n}", _time(numpy_side, 3), _time(mojo_side, 3)


def bench_scalar(iterations: int = 200000):
    """The scalar tower decision against dacite's own ``is_instance``. This is
    a loss by construction and is reported as such: one FFI call is more
    expensive than a handful of ``isinstance`` checks."""
    from dacite.types import is_instance

    pairs = [(int, 1), (float, 1.5), (complex, 1 + 0j), (bool, True),
             (float, 1), (int, 1.0)]
    for declared, value in pairs:
        assert mdac.numeric_tower_accepts(declared, value) == is_instance(
            value, declared
        )

    def dacite_side():
        for _ in range(iterations // len(pairs)):
            for declared, value in pairs:
                is_instance(value, declared)

    def mojo_side():
        for _ in range(iterations // len(pairs)):
            for declared, value in pairs:
                mdac.numeric_tower_accepts(declared, value)

    return f"is_instance scalar x{iterations}", _time(dacite_side, 3), _time(
        mojo_side, 3
    )


CASES = (
    ("column_accept_mask", bench_column),
    ("tower check mixed kinds", bench_column_rejects),
    ("cast to int", bench_cast_int),
    ("cast to bool", bench_cast_bool),
    ("is_instance scalar", bench_scalar),
)


def main():
    print(f"{'case':<30}{'reference':>13}{'mojo-dacite':>15}{'ratio':>9}")
    print("-" * 67)
    for _, fn in CASES:
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<30}{ref * 1e3:>11.2f}ms{got * 1e3:>13.2f}ms{ratio:>8.2f}x")


if __name__ == "__main__":
    main()

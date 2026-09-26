"""Parity tests for the numeric tower.

``dacite.types.is_instance`` is the reference for every case, over the full
cross product of the four builtin numeric types plus the off-tower values a
caller will actually hand a validator: ``Decimal``, ``Fraction``, ``str``,
``None``, ``numpy`` scalars, ``bool``.

The cases that matter for a plausible bug are the *asymmetric* ones. A kernel
written as ``a <= d`` gets bool/int/float/complex right but a kernel written
with the ranks the wrong way round, or one that treats ``bool`` as ``int``,
fails on exactly these.
"""

import itertools
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from dacite.types import is_instance

from mojo_dacite import column_accept_mask, kind_of, numeric_tower_accepts, pair_accept_mask

TOWER_TYPES = [bool, int, float, complex]
TOWER_VALUES = [True, False, 0, 1, -1, 2**70, 1.5, -0.0, 0.0, float("inf"),
                1 + 2j, 0j]

OFF_TOWER = [
    Decimal("1.5"), Fraction(1, 3), "1.5", "", None, b"1", object(),
    np.int64(1), np.float64(1.0), [1], (1,), {1: 2},
]


def test_tower_cross_product_matches_is_instance():
    """Every ordered pair of (declared, value) with both in the tower."""
    for declared in TOWER_TYPES:
        for value in TOWER_VALUES:
            assert numeric_tower_accepts(declared, value) == is_instance(
                value, declared
            ), (declared, value)


def test_tower_only_widens():
    """The tower accepts upwards and rejects downwards, which is the property a
    reversed rank comparison would break."""
    assert numeric_tower_accepts(int, 1)
    assert not numeric_tower_accepts(int, 1.0)
    assert not numeric_tower_accepts(int, 1 + 0j)
    assert numeric_tower_accepts(float, 1)
    assert numeric_tower_accepts(float, 1.0)
    assert not numeric_tower_accepts(float, 1 + 0j)
    assert numeric_tower_accepts(complex, 1)
    assert numeric_tower_accepts(complex, 1.0)
    assert numeric_tower_accepts(complex, 1 + 0j)
    assert numeric_tower_accepts(bool, True)
    assert not numeric_tower_accepts(bool, 1)
    assert not numeric_tower_accepts(bool, 0)


def test_bool_is_accepted_wherever_int_is():
    """bool is a subclass of int, so dacite accepts it for int, float and
    complex fields. Ranking bool as int would get this wrong in the other
    direction: it would also accept it for a bool field, which is the one place
    it is right for the wrong reason."""
    for declared in (int, float, complex):
        assert numeric_tower_accepts(declared, True) == is_instance(True, declared)
    assert not numeric_tower_accepts(bool, 1)
    assert numeric_tower_accepts(bool, 1) == is_instance(1, bool)


def test_off_tower_pairs_match_is_instance():
    for declared in TOWER_TYPES + [Decimal, Fraction, str, type(None)]:
        for value in OFF_TOWER:
            assert numeric_tower_accepts(declared, value) == is_instance(
                value, declared
            ), (declared, value)


def test_off_tower_declared_types_match_is_instance():
    for declared in (Decimal, Fraction, str, bytes, complex):
        for value in TOWER_VALUES:
            assert numeric_tower_accepts(declared, value) == is_instance(
                value, declared
            ), (declared, value)


def test_off_tower_pairs_with_each_other():
    a = [Decimal("1"), Fraction(1, 2), "x", None]
    b = [Decimal("1"), Fraction(1, 2), "x", None]
    for declared, value in itertools.product(a, b):
        assert numeric_tower_accepts(declared, value) == is_instance(value, declared)


def test_numpy_scalars_are_not_tower_members():
    """numpy int64 is not a Python int, and dacite says so. A kernel that
    keyed on duck-typed numerics would wrongly accept it."""
    for declared in TOWER_TYPES:
        for value in (np.int64(1), np.float64(1.0), np.bool_(True)):
            assert numeric_tower_accepts(declared, value) == is_instance(
                value, declared
            ), (declared, type(value))


def test_kind_of():
    assert [kind_of(t) for t in TOWER_TYPES] == [1, 2, 3, 4]
    assert kind_of(Decimal) == 0
    assert kind_of(str) == 0
    assert kind_of(np.int64) == 0


def test_column_accept_mask_matches_per_value():
    for declared in TOWER_TYPES + [Decimal, Fraction, str]:
        for values in (
            TOWER_VALUES,
            OFF_TOWER,
            TOWER_VALUES + OFF_TOWER,
            [],
            [True] * 5,
        ):
            got = column_accept_mask(declared, values)
            expect = [is_instance(v, declared) for v in values]
            assert got.tolist() == [bool(e) for e in expect], (declared, values)


def test_column_accept_mask_is_ordered_and_complete():
    """A mask that was the right length but permuted, or off by one, would show
    up here because the values are not symmetric."""
    values = [0, 1.0, 2, 3.5, "x", True, None, 4]
    got = column_accept_mask(float, values)
    expect = [is_instance(v, float) for v in values]
    assert got.tolist() == expect
    assert got.dtype == bool


def test_pair_accept_mask_matches_per_pair():
    declared = [int, float, complex, bool, Decimal, str, int]
    actual = [1, 1, 1, 1, 1, 1, 1]
    got, rejected = pair_accept_mask(declared, actual)
    expect = [is_instance(v, d) for d, v in zip(declared, actual)]
    assert got.tolist() == expect
    assert rejected == sum(1 for e in expect if not e)


def test_pair_accept_mask_large_and_mixed():
    rng = np.random.default_rng(4)
    pool = TOWER_TYPES + [Decimal, Fraction, str]
    values = TOWER_VALUES + OFF_TOWER
    declared = [pool[i] for i in rng.integers(0, len(pool), size=5000)]
    actual = [values[i] for i in rng.integers(0, len(values), size=5000)]
    got, rejected = pair_accept_mask(declared, actual)
    expect = [is_instance(v, d) for d, v in zip(declared, actual)]
    assert got.tolist() == expect
    assert rejected == sum(1 for e in expect if not e)


def test_pair_accept_mask_length_mismatch():
    with pytest.raises(ValueError):
        pair_accept_mask([int], [1, 2])


def test_pair_accept_mask_empty():
    got, rejected = pair_accept_mask([], [])
    assert got.size == 0 and rejected == 0

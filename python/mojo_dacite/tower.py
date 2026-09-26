"""The PEP 484 numeric tower, which is the whole of dacite's numeric surface.

dacite checks every field value with ``dacite.types.is_instance``. Almost all
of that function is ``typing`` introspection. The one arithmetic-shaped line in
all of dacite is its numeric tower:

    if (type_ in [float, complex] and isinstance(value, (int, float))) or isinstance(value, type_):

That is a widening rule over a rank ordering -- ``bool < int < float <
complex`` -- so it reduces to one integer comparison. Here it is that
comparison, scalar and in bulk.

The bulk entry point exists for the case dacite itself does not have: many
rows of one schema. dacite validates a record at a time, so a caller checking a
table calls ``is_instance`` once per row; ``column_accept_mask`` makes the same
decision for a whole column in one call.

Anything off the tower -- a ``Decimal`` field, a ``str`` value, a
``numpy.int64`` -- is forwarded to the real ``dacite.types.is_instance``,
because the kernel only carries the four builtin kinds and guessing at the rest
would be wrong rather than merely incomplete.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import _lib
from ._lib import KIND_NAMES

__all__ = [
    "KIND_NAMES",
    "TOWER_TYPES",
    "cast_numeric",
    "cast_numeric_array",
    "column_accept_mask",
    "kind_of",
    "numeric_tower_accepts",
    "pair_accept_mask",
]

# The four builtin types the tower is defined over, in widening order.
TOWER_TYPES = [bool, int, float, complex]

# Python type -> tower kind. bool is registered alongside int deliberately:
# bool is a subclass of int, and dacite's tower accepts a bool for an int
# field exactly as isinstance does, so it must rank *below* int rather than be
# conflated with it.
_TYPE_KINDS = {
    bool: _lib.KIND_BOOL,
    int: _lib.KIND_INT,
    float: _lib.KIND_FLOAT,
    complex: _lib.KIND_COMPLEX,
}


def kind_of(type_: Any) -> int:
    """Tower kind of a Python type; ``KIND_OTHER`` for anything off the tower.

    Off-tower includes ``Decimal`` and ``Fraction`` (dacite accepts those only
    as an exact type match, which is not a rank), and any subclass of a tower
    type other than ``bool`` -- the kernel does not know about those, so the
    caller falls back to the real ``is_instance``.
    """
    return _TYPE_KINDS.get(type_, _lib.KIND_OTHER)


def _off_tower_fixup(declared, actual, mask: np.ndarray, indices) -> None:
    from dacite.types import is_instance

    for i in indices:
        mask[i] = bool(is_instance(actual[i], declared[i]))


def numeric_tower_accepts(declared: Any, value: Any) -> bool:
    """``dacite.types.is_instance(value, declared)``.

    Exact for every pair of builtin numeric types, and for any pair involving
    an off-tower type or value it defers to the real implementation, so the
    answer is always dacite's.
    """
    from dacite.types import is_instance

    d = kind_of(declared)
    a = kind_of(type(value))
    if d == _lib.KIND_OTHER or a == _lib.KIND_OTHER:
        return bool(is_instance(value, declared))
    return bool(_lib.lib.dac_tower_accepts(d, a))


def pair_accept_mask(
    declared: list[Any], actual: list[Any]
) -> tuple[np.ndarray, int]:
    """Tower decision for aligned runs of declared types and values.

    Returns ``(mask, rejected)`` where ``mask[i]`` is 1 when dacite would
    accept ``actual[i]`` for ``declared[i]``. Any pair where either side is off
    the tower is resolved individually through the real ``is_instance``, so the
    kernel handles the runs of genuine tower members and Python handles the
    rest.
    """
    n = len(declared)
    if len(actual) != n:
        raise ValueError("declared and actual must be the same length")
    d_kinds = np.array([kind_of(t) for t in declared], dtype=np.int32)
    a_kinds = np.array([kind_of(type(v)) for v in actual], dtype=np.int32)
    out = np.empty(n, dtype=np.uint8)
    if n:
        _lib.lib.dac_tower_check(
            d_kinds.ctypes.data, a_kinds.ctypes.data, n, out.ctypes.data
        )
    mask = out.astype(bool)
    off = np.flatnonzero(
        (d_kinds == _lib.KIND_OTHER) | (a_kinds == _lib.KIND_OTHER)
    )
    if off.size:
        _off_tower_fixup(declared, actual, mask, off)
    return mask, n - int(mask.sum())


def column_accept_mask(declared: Any, values, *, kinds=None) -> np.ndarray:
    """Accept/reject a whole column of values against one annotated type.

    This is the useful shape for a table: dacite's own loop calls
    ``is_instance`` once per row, which for a million rows is a million
    interpreter round trips. Here the tower decision is one pass over a
    contiguous int32 buffer.

    ``kinds`` lets a caller that has already classified the column pass the
    per-value tower ranks in directly. That classification -- one ``type()``
    call and one dict lookup per value -- is Python work the kernel cannot do,
    and it is usually the larger half of the wall clock, so it is worth being
    able to hoist out of a loop over columns.

    Returns a boolean array, one entry per value, in order.
    """
    from dacite.types import is_instance

    d = kind_of(declared)
    if d == _lib.KIND_OTHER:
        return np.array(
            [bool(is_instance(v, declared)) for v in values], dtype=bool
        )
    if kinds is None:
        kinds = np.array(
            [kind_of(type(v)) for v in values], dtype=np.int32
        )
    else:
        kinds = np.ascontiguousarray(kinds, dtype=np.int32)
    out = np.empty(kinds.size, dtype=np.uint8)
    if kinds.size:
        _lib.lib.dac_tower_check_one(
            d, kinds.ctypes.data, kinds.size, out.ctypes.data
        )
    mask = out.astype(bool)
    off = np.flatnonzero(kinds == _lib.KIND_OTHER)
    for i in off:
        mask[i] = bool(is_instance(values[int(i)], declared))
    return mask


def cast_numeric(values, target: type) -> list:
    """dacite's ``Config.cast`` coercion for numeric targets, over a run.

    dacite's cast is ``data = type_(data)``, and for ``int``/``float``/``bool``
    that is arithmetic: truncation toward zero, the identity, and the
    zero/non-zero collapse. The kernel does it for the whole run.

    Three things it deliberately does not do, all stated rather than hidden:

    * ``complex`` is not handled -- the value would have to cross the ABI as a
      complex128 pair, and a caller wanting one complex value is better served
      by Python's own ``complex()`` than by a kernel round trip. ``complex`` is
      passed straight through, so the answer is still dacite's.
    * an ``int`` wider than 2**53 is not exactly representable in float64, and
      routing one through the kernel would answer ``int(2**53 + 1)`` with
      ``2**53``. Such values are computed by Python instead, so
      ``cast_numeric([2**53 + 1], int)`` is exact. This is the only way a
      float64 kernel can be honest about an unbounded Python ``int``.
    * a float outside int64 is flagged by the kernel and recomputed with
      Python's unbounded ``int``, and so is NaN and the infinities. That is
      ``int(2**64)`` succeeding and ``int(float("nan"))`` raising, which is
      exactly what dacite does.
    """
    targets = {int: _lib.CAST_INT, float: _lib.CAST_FLOAT, bool: _lib.CAST_BOOL}
    if target not in targets:
        return [target(v) for v in values]

    n = len(values)
    result: list = [None] * n
    # Values the kernel can represent exactly, and the indices they came from.
    run: list = []
    run_idx: list = []
    for i, v in enumerate(values):
        if isinstance(v, int) and not (-(2**53) <= v <= 2**53):
            result[i] = target(v)
        else:
            run.append(v)
            run_idx.append(i)

    arr = np.ascontiguousarray(run, dtype=np.float64)
    out = np.empty(arr.size, dtype=np.float64)
    flags = np.empty(arr.size, dtype=np.int32)
    if arr.size:
        _lib.lib.dac_cast_numeric(
            arr.ctypes.data, arr.size, targets[target],
            out.ctypes.data, flags.ctypes.data,
        )
    # tolist() rather than indexing element by element: reading a NumPy scalar
    # costs about a microsecond, which would be most of this function's time.
    out_list = out.tolist()
    flag_list = flags.tolist()
    src_list = arr.tolist()
    ok = _lib.FLAG_OK
    for j, i in enumerate(run_idx):
        if flag_list[j] == ok:
            v = out_list[j]
            result[i] = int(v) if target is int else (
                v if target is float else bool(v)
            )
        else:
            # FLAG_RANGE or FLAG_UNSUPPORTED: let Python produce the value or
            # the exception, so both are upstream's.
            result[i] = target(src_list[j])
    return result


def cast_numeric_array(values, target: type) -> np.ndarray:
    """``cast_numeric`` for a run that stays an array.

    ``cast_numeric`` returns a Python list, which for a million values is a
    million boxed objects and dominates everything the kernel does. When the
    caller wants an array, this is the same kernel call with the result left as
    one, and it is the honest shape to benchmark.

    ``int`` returns int64, ``float`` returns float64, ``bool`` returns bool.
    A value the kernel flags -- a float outside int64, NaN, an infinity --
    raises here exactly as it would in the scalar path, because there is no
    per-element fallback in an array result. Use ``cast_numeric`` when the run
    can contain those.
    """
    targets = {int: _lib.CAST_INT, float: _lib.CAST_FLOAT, bool: _lib.CAST_BOOL}
    if target not in targets:
        return np.array([target(v) for v in values])
    arr = np.ascontiguousarray(values, dtype=np.float64)
    out = np.empty(arr.size, dtype=np.float64)
    flags = np.empty(arr.size, dtype=np.int32)
    if arr.size:
        _lib.lib.dac_cast_numeric(
            arr.ctypes.data, arr.size, targets[target],
            out.ctypes.data, flags.ctypes.data,
        )
    bad = np.flatnonzero(flags != _lib.FLAG_OK)
    if bad.size:
        i = int(bad[0])
        raise OverflowError(
            f"cast to {target.__name__} is not representable for element {i}: "
            f"{float(arr[i])!r}"
        )
    if target is int:
        return out.astype(np.int64)
    if target is bool:
        return out != 0.0
    return out

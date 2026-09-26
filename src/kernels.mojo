"""Mojo kernels for the numeric surface of dacite.

dacite builds a dataclass instance from a mapping and checks each value against
the field's annotation. It is a type-dispatch library: 689 lines of
``isinstance``/``typing.get_origin`` recursion, and not one loop over numbers
or bytes. This port does not manufacture a numeric core that upstream does not
have.

There is exactly one place where dacite makes a decision that is arithmetic
rather than type introspection, and it is the whole of this file:

    # As described in PEP 484 - section: "The numeric tower"
    if (type_ in [float, complex] and isinstance(value, (int, float))) or isinstance(value, type_):

That single expression is the numeric tower: ``float`` admits an ``int``,
``complex`` admits an ``int`` and a ``float``, and nothing is admitted
downward. Because the tower is a *rank* ordering -- bool < int < float <
complex -- the whole rule is one integer comparison per pair, which vectorises
over a column of values in a way the Python expression cannot.

The second exported kernel is the other half of the same story: dacite's
``Config.cast`` coercion, ``data = type_(data)``, for the numeric targets. That
one does arithmetic -- truncation toward zero, the bool collapse, and the
int64 representability check -- and it has a real correctness hazard worth
naming, namely that a large ``int`` is not exact in float64.

Every exported symbol takes buffer addresses as plain ``Int`` values and
rebuilds the pointer inside the body, because ``@export`` rejects parametric
functions and an inferred pointer origin would make the symbol parametric.
"""

from std.math import floor as _floor

comptime U8P = Pointer[UInt8, AnyOrigin[mut=True]]
comptime I32P = Pointer[Int32, AnyOrigin[mut=True]]
comptime F64P = Pointer[Float64, AnyOrigin[mut=True]]

# Ranks in the numeric tower. The ordering is the whole rule: a value is
# accepted for an annotated type when its rank is at or below that type's rank
# and the value is itself a member of the tower.
comptime KIND_OTHER: Int32 = 0
comptime KIND_BOOL: Int32 = 1
comptime KIND_INT: Int32 = 2
comptime KIND_FLOAT: Int32 = 3
comptime KIND_COMPLEX: Int32 = 4

# Cast targets for dac_cast_numeric.
comptime CAST_INT: Int32 = 0
comptime CAST_FLOAT: Int32 = 1
comptime CAST_BOOL: Int32 = 2
comptime CAST_COMPLEX: Int32 = 3

# Out-of-range flags from dac_cast_numeric.
comptime FLAG_OK: Int32 = 0
comptime FLAG_RANGE: Int32 = 1
comptime FLAG_UNSUPPORTED: Int32 = 2


@export("dac_tower_accepts")
def dac_tower_accepts(declared: Int32, actual: Int32) abi("C") -> Int32:
    """The PEP 484 numeric tower for one (declared, actual) pair.

    Returns 1 when dacite's ``is_instance(value, type_)`` would be true for
    these two kinds. ``KIND_OTHER`` is a declared or actual type outside the
    tower -- ``Decimal``, ``Fraction``, a string -- and never matches anything
    but itself, which for a non-numeric type is handled in Python where the
    real type object exists.

    Note the direction: an ``int`` annotated field rejects a ``float`` even
    though the number fits, because the tower only widens. And ``bool`` is a
    subclass of ``int`` in Python, so it is accepted for ``int``, ``float``
    and ``complex`` -- which is upstream's behaviour, not a rounding of it.
    """
    if declared == KIND_OTHER or actual == KIND_OTHER:
        return Int32(1) if declared == actual else Int32(0)
    if actual <= declared:
        return Int32(1)
    return Int32(0)


@export("dac_tower_check")
def dac_tower_check(
    declared_addr: Int, actual_addr: Int, n: Int, out_addr: Int
) abi("C") -> Int64:
    """The tower rule over ``n`` (declared, actual) pairs in one pass.

    ``out[i]`` is 1 when the pair is accepted. Returns the number of
    rejections, which is what a caller validating a whole column wants to know
    before looking at any individual row.

    ``declared`` and ``actual`` are runs of int32 matched by position -- the
    same position that maps a field name to its column. A rejected pair is a
    validation failure, not an error. The mask is one byte per pair, not four,
    because at these sizes the output traffic is what the kernel is bound by.
    """
    var declared = I32P(unsafe_from_address=declared_addr)
    var actual = I32P(unsafe_from_address=actual_addr)
    var out = U8P(unsafe_from_address=out_addr)
    var bad = Int64(0)
    for i in range(n):
        var d = declared[unsafe_offset=i]
        var a = actual[unsafe_offset=i]
        var ok = a <= d
        if d == KIND_OTHER or a == KIND_OTHER:
            ok = d == a
        if ok:
            out[unsafe_offset=i] = UInt8(1)
        else:
            out[unsafe_offset=i] = UInt8(0)
            bad += Int64(1)
    return bad


@export("dac_tower_check_one")
def dac_tower_check_one(
    declared: Int32, actual_addr: Int, n: Int, out_addr: Int
) abi("C") -> Int64:
    """The tower rule for one declared type against a run of values.

    This is the column shape, and it is the one worth having as its own export:
    a table of records has one annotation per field and many rows, so the
    declared rank is a scalar here and the kernel reads a single int32 array
    instead of two. Half the memory traffic of ``dac_tower_check``.

    ``out[i]`` is 1 when the value's kind is accepted. Returns the number of
    rejections. ``KIND_OTHER`` declared means "not a tower type" and is
    rejected for every value, which is the caller's cue to defer to the real
    ``is_instance``.
    """
    var actual = I32P(unsafe_from_address=actual_addr)
    var out = U8P(unsafe_from_address=out_addr)
    var bad = Int64(0)
    if declared == KIND_OTHER:
        # Not a tower type: only an exact match is accepted, and the caller's
        # cue to defer to the real is_instance.
        for i in range(n):
            if actual[unsafe_offset=i] == KIND_OTHER:
                out[unsafe_offset=i] = UInt8(1)
            else:
                out[unsafe_offset=i] = UInt8(0)
                bad += Int64(1)
        return bad
    for i in range(n):
        var a = actual[unsafe_offset=i]
        var ok = a <= declared
        if a == KIND_OTHER:
            ok = False
        if ok:
            out[unsafe_offset=i] = UInt8(1)
        else:
            out[unsafe_offset=i] = UInt8(0)
            bad += Int64(1)
    return bad


@export("dac_cast_numeric")
def dac_cast_numeric(
    src_addr: Int, n: Int, target: Int32, out_addr: Int, flag_addr: Int
) abi("C") -> Int64:
    """The ``Config.cast`` coercion for numeric targets, over a run.

    dacite's cast is literally ``data = type_(data)``, and for the numeric types
    that is:

    * ``int(v)`` -- truncation toward zero, exact for any finite float64;
    * ``float(v)`` -- the identity on a float64 input;
    * ``bool(v)`` -- 0 when ``v`` is zero, 1 otherwise, including for NaN;
    * ``complex(v)`` -- rejected here (``FLAG_UNSUPPORTED``) because the value
      would have to cross the ABI as a complex128 pair and the caller can call
      Python's ``complex()`` for a single token far more cheaply than this
      kernel can be told about it.

    The hazard worth naming: an ``int`` target has no range limit in Python,
    but this kernel stores into an int64 result and so flags anything outside
    ``[-2**63, 2**63)`` with ``FLAG_RANGE`` instead of silently wrapping. The
    shim recomputes those with Python's unbounded ``int``. ``NaN`` and the
    infinities are outside every Python ``int``, and are flagged too.

    Returns the number of flagged elements.
    """
    var src = F64P(unsafe_from_address=src_addr)
    var out = F64P(unsafe_from_address=out_addr)
    var flag = I32P(unsafe_from_address=flag_addr)
    var flagged = Int64(0)
    var limit = Float64(9223372036854775808.0)
    for i in range(n):
        var v = src[unsafe_offset=i]
        if target == CAST_INT:
            if v != v or v >= limit or v < -limit:
                out[unsafe_offset=i] = Float64(0.0)
                flag[unsafe_offset=i] = FLAG_RANGE
                flagged += Int64(1)
            else:
                # Truncation toward zero, matching int(float).
                out[unsafe_offset=i] = _trunc(v)
                flag[unsafe_offset=i] = FLAG_OK
        elif target == CAST_BOOL:
            out[unsafe_offset=i] = Float64(0.0) if v == Float64(0.0) else Float64(1.0)
            flag[unsafe_offset=i] = FLAG_OK
        elif target == CAST_FLOAT:
            out[unsafe_offset=i] = v
            flag[unsafe_offset=i] = FLAG_OK
        else:
            out[unsafe_offset=i] = v
            flag[unsafe_offset=i] = FLAG_UNSUPPORTED
            flagged += Int64(1)
    return flagged


@inline(.always)
def _trunc(v: Float64) -> Float64:
    """Truncation toward zero, matching Python's ``int(float)``.

    ``floor`` rounds toward negative infinity, so for a negative input the
    answer is ``-floor(|v|)`` -- ``int(-1.5)`` is -1, not -2. There is no
    fractional-part correction to make: ``floor`` of a positive magnitude is
    already the truncation of that magnitude.
    """
    if v < Float64(0.0):
        return -_floor(-v)
    return _floor(v)

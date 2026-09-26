"""mojo-dacite: dacite's numeric surface, with Mojo kernels.

dacite builds a dataclass instance from a mapping and checks each value against
the field's annotation. It is 689 lines of ``typing`` introspection and
recursive dispatch, and it contains no loop over numbers or bytes. This package
does not pretend otherwise.

What it does port is the one place dacite makes an arithmetic rather than a
type-introspection decision: the PEP 484 numeric tower, plus the numeric half
of ``Config.cast``'s coercion. The tower is a rank comparison -- ``bool <
int < float < complex``, accepting only upwards -- and both directions are
kernels here: one comparison for a single pair, one pass for a column.

Everything else dacite does -- the union and generic-collection walks, the
dataclass recursion, the forward-reference resolution, the exception hierarchy
and its path tracking -- is not ported. Use the real `dacite` for that.
"""

from .tower import (
    KIND_NAMES,
    TOWER_TYPES,
    cast_numeric,
    cast_numeric_array,
    column_accept_mask,
    kind_of,
    numeric_tower_accepts,
    pair_accept_mask,
)

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
__version__ = "0.1.0"

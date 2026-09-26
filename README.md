# mojo-dacite

The compute-oriented subset of [dacite](https://github.com/konradhalas/dacite),
with its numeric decisions in Mojo.

## dacite has no numeric core, and this README says so first

dacite is 689 lines of `typing` introspection and recursive dispatch. It
builds a dataclass instance from a mapping and checks each value against the
field's annotation. It contains **no loop over numbers and no loop over
bytes**. There is no array processing in it to speed up, and this port does
not manufacture one.

There is exactly one place in all of dacite where a decision is arithmetic
rather than type introspection, and that is the whole of this project:

```python
# As described in PEP 484 - section: "The numeric tower"
if (type_ in [float, complex] and isinstance(value, (int, float))) or isinstance(value, type_):
```

That is `dacite.types.is_instance`'s numeric tower: `float` admits an `int`,
`complex` admits an `int` and a `float`, and nothing admits downward. Because
it is a *widening rule over a rank ordering* — `bool < int < float < complex` —
the whole expression collapses to one integer comparison, which is what the
kernels here are. The second half is the numeric part of `Config.cast`,
where dacite's coercion is `data = type_(data)` and `int(v)` is a real
truncation.

The Python package is `mojo_dacite`, so it installs alongside the real
`dacite`, and the parity tests import both.

```python
import mojo_dacite as mdac

mdac.numeric_tower_accepts(float, 1)        # True -- int widens to float
mdac.numeric_tower_accepts(int, 1.0)        # False -- never narrows
mdac.column_accept_mask(float, [1, 2.0, "x"])   # array([True, True, False])
mdac.cast_numeric([-1.5, 2.7], int)         # [-1, 2] -- toward zero
```

## Covered subset

| area | ported API | what the kernel does |
| --- | --- | --- |
| Numeric tower, one pair | `numeric_tower_accepts` | `dac_tower_accepts`: one rank comparison, the whole of dacite's numeric rule |
| Numeric tower, a column | `column_accept_mask` | `dac_tower_check_one`: one declared rank against a run of value ranks, byte mask out |
| Numeric tower, paired | `pair_accept_mask` | `dac_tower_check`: aligned runs of declared and actual ranks |
| `Config.cast`, numeric targets | `cast_numeric`, `cast_numeric_array` | `dac_cast_numeric`: truncation toward zero, the bool collapse, the int64 range check |

Not implemented, and not ported: `dacite.from_dict` itself, the union walk
(`_build_value_for_union` and `StrictUnionMatchError`), the generic-collection
walk, the nested-dataclass recursion, forward-reference resolution, the
`Config` type-hook and key-conversion machinery, and the whole
`WrongTypeError` / `MissingValueError` / `UnexpectedDataError` /
`DaciteFieldError` hierarchy with its path tracking. All of that is control
flow and string work. Use the real `dacite` to validate a dict into a
dataclass; this package answers the numeric questions it would ask.

`dacite.types.is_instance` for a non-numeric pair, and any off-tower value
(`Decimal`, `Fraction`, `str`, `numpy` scalars, a subclass), is **forwarded to
the real `dacite.types.is_instance`**. The kernel carries only the four
builtin kinds, and guessing at the rest would be wrong rather than merely
incomplete.

### Deliberate delegations

- `complex` is not cast in the kernel. The value would have to cross the ABI
  as a complex128 pair, and a caller wanting one complex value is better served
  by Python's own `complex()`. `cast_numeric(values, complex)` passes straight
  through, so the answer is still dacite's.
- An `int` wider than 2**53 is not exactly representable in float64, and
  routing one through the kernel would answer `int(2**53 + 1)` with `2**53`.
  Such values are computed by Python instead, so the result is exact.
- A float outside int64 is flagged by the kernel and recomputed with Python's
  unbounded `int`. That is `int(2**64)` succeeding and `int(float("nan"))`
  raising, which is what dacite does.

## Install

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` compiles `src/kernels.mojo` into
`dist/libmojo-dacite.so`. Set `PYTHONPATH=python` outside a Pixi task.
`pixi run bench` runs the benchmark below.

`pytest.ini` in the project root disables the `zarr` pytest plugin. The shared
test venv pairs zarr 3.4.0 with numpy 1.26 and `zarr.testing` fails to import,
and pytest loads entry-point plugins before it reads `conftest.py`, so a
project-level file is the only place that can stop it. Nothing here uses zarr.

## Performance

Best-of-three wall clock, one process. Every case checks agreement with
`dacite.types.is_instance` or with Python's own conversion first. The box is
shared, so single runs vary by 2-3x; these are one representative run.

| case | reference | mojo-dacite | result |
| --- | ---: | ---: | ---: |
| `column_accept_mask`, 4.2M values, vs the fastest NumPy formulation of the whole operation | 782.31 ms | 47.00 ms | 16.6x faster |
| tower check, 4.2M mixed kinds, vs `rank <= 3` | 2.79 ms | 10.10 ms | 0.28x — 3.6x slower |
| cast to int, 4.2M, vs `np.trunc` | 192.24 ms | 151.13 ms | 1.27x faster |
| cast to bool, 4.2M, vs `arr != 0` | 7.71 ms | 55.47 ms | 0.14x — 7.2x slower |
| `is_instance`, 200k scalar calls, vs dacite | 165.43 ms | 1040.37 ms | 0.16x — 6.3x slower |

Read these honestly:

- **`column_accept_mask` is the case the kernel is for.** The reference
  includes classifying each value's Python type, which is a `type()` call and a
  dict lookup per value and cannot be vectorised at all. The kernel does the
  decision; the classification is Python either way, which is why the shim
  accepts precomputed `kinds` and why the benchmark's fast Mojo number is with
  the classification hoisted out.
- **The raw tower check loses to NumPy.** `rank <= 3` is one NumPy pass over
  16 MB; the kernel reads the same 16 MB and writes a 4 MB byte mask with a
  per-element `KIND_OTHER` test that blocks vectorisation. Splitting the
  off-tower case into its own loop took it from 29 ms to 10 ms, and there is
  not much left to win at this size.
- **The cast to int is a modest win** because the kernel avoids materialising
  NumPy temporaries. The cast to bool loses: `arr != 0.0` is a single
  vectorised pass, and the kernel additionally writes a value array the caller
  does not need.
- **The scalar case is a loss by construction.** One FFI call is more
  expensive than dacite's handful of `isinstance` checks, and there is no
  batching available for a single pair. It is on the list because omitting it
  would flatter the port.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit. Buffers cross
the C ABI as 64-bit addresses and are reconstructed in Mojo as
`Pointer[T, AnyOrigin[mut=True]]`, which keeps the exported symbols
non-parametric — `@export` rejects a parametric function, and an inferred
pointer origin would make it one.

The single-declared-kind export exists as a separate symbol from the paired
one for a concrete reason: a table of records has one annotation per field and
many rows, so the declared rank is a scalar and the kernel reads one int32
array instead of two. That is half the memory traffic, and it measured 29 ms
against 10 ms before the off-tower branch was split out.

Nothing here is floating-point arithmetic, so **exact** equality is the right
assertion and the tests use it. There is no FMA concern: the cast's only
floating-point work is a `floor`, which is exact.

## Tests

```
28 passed
```

- `tests/test_tower.py` — `is_instance` parity over the full cross product of
  the four numeric types and twelve representative values, plus off-tower
  types, `numpy` scalars and a 5000-pair random mix. The asymmetric cases carry
  the weight: a reversed rank comparison fails exactly there.
- `tests/test_cast.py` — the cast against Python's own conversion, with
  truncation toward zero called out separately because floor instead of trunc
  is the obvious kernel bug, and with the int-beyond-float64 and
  int-beyond-int64 boundaries called out because both are silent-wrong failures.

## License

MIT

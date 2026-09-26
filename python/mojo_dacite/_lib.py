"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory: every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay ``c_int64`` for addresses. ``c_int``
truncates them and segfaults.
"""

import ctypes
import pathlib

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-dacite.so"

I64 = ctypes.c_int64
I32 = ctypes.c_int32
F64 = ctypes.c_double


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    lib.dac_tower_accepts.restype = I32
    lib.dac_tower_accepts.argtypes = [I32, I32]

    lib.dac_tower_check.restype = I64
    lib.dac_tower_check.argtypes = [I64, I64, I64, I64]

    lib.dac_tower_check_one.restype = I64
    lib.dac_tower_check_one.argtypes = [I32, I64, I64, I64]

    lib.dac_cast_numeric.restype = I64
    lib.dac_cast_numeric.argtypes = [I64, I64, I32, I64, I64]
    return lib


lib = _load()

# Kinds, in tower order. The ordering is the rule: a value is accepted for an
# annotated type when its kind is at or below that type's kind. Kept in step
# with the KIND_* constants in src/kernels.mojo.
KIND_OTHER = 0
KIND_BOOL = 1
KIND_INT = 2
KIND_FLOAT = 3
KIND_COMPLEX = 4

KIND_NAMES = {
    KIND_OTHER: "other",
    KIND_BOOL: "bool",
    KIND_INT: "int",
    KIND_FLOAT: "float",
    KIND_COMPLEX: "complex",
}

# Cast targets, in step with the CAST_* constants in src/kernels.mojo.
CAST_INT = 0
CAST_FLOAT = 1
CAST_BOOL = 2
CAST_COMPLEX = 3

# Out-of-range flags, in step with the FLAG_* constants.
FLAG_OK = 0
FLAG_RANGE = 1
FLAG_UNSUPPORTED = 2

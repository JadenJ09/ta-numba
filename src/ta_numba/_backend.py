"""Rust backend detection and dispatch helpers."""

import os

_RUST_AVAILABLE = False
_rs = None

if not os.environ.get("TA_NUMBA_DISABLE_RUST"):
    try:
        from . import _ta_numba_rs
        _RUST_AVAILABLE = True
        _rs = _ta_numba_rs
    except ImportError:
        pass


def get_backend() -> str:
    """Return the active computation backend: 'rust' or 'numba'."""
    return "rust" if _RUST_AVAILABLE else "numba"


def is_rust_available() -> bool:
    """Check if the Rust acceleration backend is available."""
    return _RUST_AVAILABLE

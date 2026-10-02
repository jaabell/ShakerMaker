# FFSP (Finite Fault Stochastic Process) compiled Fortran extension.
# The public API is exposed through FFSPSource in shakermaker.ffspsource.
# This module exposes only the low-level compiled wrapper when available.

try:
    from shakermaker.ffsp.ffsp_core import ffsp_run_wrapper
    _ffsp_available = True
except ImportError:
    _ffsp_available = False

__all__ = ["ffsp_run_wrapper"] if _ffsp_available else []
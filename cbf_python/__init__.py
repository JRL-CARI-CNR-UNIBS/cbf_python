"""
CBF Python Package: Modular, high-performance Control Barrier Functions for robot manipulators.
"""

import sys
import os
from pathlib import Path

# Automatically ensure CMEEL libraries from virtual environment have precedence over conflicting system paths
_venv_cmeel_py = Path("/home/galileo/projects/python_venv/galileo_venv/lib/python3.12/site-packages/cmeel.prefix/lib/python3.12/site-packages")
if _venv_cmeel_py.exists():
    _venv_cmeel_str = str(_venv_cmeel_py)
    if _venv_cmeel_str in sys.path:
        sys.path.remove(_venv_cmeel_str)
    sys.path.insert(0, _venv_cmeel_str)

_venv_cmeel_lib = Path("/home/galileo/projects/python_venv/galileo_venv/lib/python3.12/site-packages/cmeel.prefix/lib")
if _venv_cmeel_lib.exists():
    _cmeel_lib_str = str(_venv_cmeel_lib)
    _ld_parts = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    if _ld_parts[0] != _cmeel_lib_str and not os.environ.get("_CMEEL_REEXEC"):
        os.environ["LD_LIBRARY_PATH"] = f"{_cmeel_lib_str}:{os.environ.get('LD_LIBRARY_PATH', '')}".rstrip(":")
        os.environ["_CMEEL_REEXEC"] = "1"
        try:
            if hasattr(sys, "orig_argv") and sys.orig_argv:
                os.execv(sys.orig_argv[0], sys.orig_argv)
            else:
                os.execv(sys.executable, [sys.executable] + sys.argv)
        except Exception:
            pass


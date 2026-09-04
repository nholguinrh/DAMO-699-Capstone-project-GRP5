"""
Repo-root entry point for the executive forecasting dashboard (Issue #104).

Launch from the project root with:

    streamlit run streamlit_app.py

This shim puts the project root and ``src/dashboard`` on ``sys.path`` so the
dashboard can import both its own modules (``data_loader``, ``charts``) and the
shared pipeline code (``src.pipeline_runner`` and friends), then runs the real
app at ``src/dashboard/app.py``. Running that file directly still works too.
"""

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_APP = _ROOT / "src" / "dashboard" / "app.py"

for _p in (str(_ROOT), str(_ROOT / "src" / "dashboard")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

runpy.run_path(str(_APP), run_name="__main__")

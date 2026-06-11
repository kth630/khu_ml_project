"""
Report pipeline step 02: Exploratory data analysis.

This file is intentionally a thin report-friendly entry point.
The original implementation is kept in A_ml_eda.py so existing code is not deleted or changed.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    script_path = Path(__file__).resolve().parent / "A_ml_eda.py"
    runpy.run_path(str(script_path), run_name="__main__")

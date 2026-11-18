"""Run the whole project from the raw downloads to the figures and result tables.

Usage:  python run_all.py
Takes roughly 20 to 30 minutes on a laptop (most of it is the rolling SARIMAX and VAR refits).
"""
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).parent / "src"
STEPS = ["data_prep.py", "s01_eda.py", "s02_daily_models.py", "s03_var_granger.py", "s04_ml_models.py",
         "s05_hourly.py", "s06_extras.py"]

for step in STEPS:
    start = time.time()
    print(f"\n===== {step} =====", flush=True)
    subprocess.run([sys.executable, step], cwd=SRC, check=True)
    print(f"({step} finished in {time.time() - start:.0f} s)", flush=True)

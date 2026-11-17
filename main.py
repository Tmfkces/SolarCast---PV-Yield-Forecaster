"""Launch the PV Yield Forecaster Streamlit app from the repository root.

    python main.py            # same as: cd pv-yield-pk && streamlit run app.py
"""

import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent / "pv-yield-pk"


def main() -> int:
    command = [sys.executable, "-m", "streamlit", "run", "app.py", *sys.argv[1:]]
    return subprocess.call(command, cwd=APP_DIR)


if __name__ == "__main__":
    raise SystemExit(main())

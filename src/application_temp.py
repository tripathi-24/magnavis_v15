"""
Backward-compatible entry point. The primary module is `app.py`.

Run:  python src/app.py
Same: python src/application_temp.py
"""
from __future__ import annotations

import pathlib
import runpy
import sys


def main() -> None:
    app_path = pathlib.Path(__file__).resolve().parent / "app.py"
    sys.argv[0] = str(app_path)
    runpy.run_path(str(app_path), run_name="__main__")


if __name__ == "__main__":
    main()

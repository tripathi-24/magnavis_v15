#!/usr/bin/env python3
"""Thin entrypoint so ``python run_suite.py`` runs the improved driver (not the parent folder)."""
from __future__ import annotations

import run_suite_improved

if __name__ == "__main__":
    run_suite_improved.main()

# Thesis assets (optional)

Use this folder for material you reference directly from your LaTeX thesis (figure copies, notation notes). It is optional; canonical experiment outputs remain under:

`src/benchmark_feb13_2026_improved/results/k_recall_curves_zero_hist_20260522_170707/`

Suggested workflow:

1. Copy or symlink thesis figures from that results bundle into `thesis/figures/`.
2. `\includegraphics{figures/k_recall_curves_five_scenarios}` from your main `.tex` file if the thesis repo root is this project.

Do not duplicate multi-megabyte CSVs here; keep them in `Datafiles/` (local only).

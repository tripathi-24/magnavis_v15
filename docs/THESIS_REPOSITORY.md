# Thesis repository guide (Magnavis v15)

This document describes how to publish a **lean, reproducible** Git repository for thesis work without pushing multi-gigabyte CSVs or benchmark session trees.

## What belongs on GitHub

| Track in Git | Keep local only |
|--------------|-----------------|
| `src/` application, predictors, detectors | `Datafiles/*.csv` (see `Datafiles/README.md`) |
| `docs/` canonical script documentation | `src/sessions/`, `snapshots/` |
| `tools/` evaluation utilities | Benchmark `results/**/runs/` (app logs, eval trees) |
| `models/` bundled LSTM/GRU checkpoints (~9 MB total) | Dec-2025 1.5 GB training CSV |
| Benchmark **summaries**: `*.md`, `*.csv`, `*.png`, `*.tex` at bundle roots | `nohup.log`, training logs, ablation `results/**/models/` |
| `requirements_feb_2025.txt` | Full PDF papers (thesis folder on disk) |

**Canonical thesis results bundle** (figures + tables for the write-up):

`src/benchmark_feb13_2026_improved/results/k_recall_curves_zero_hist_20260522_170707/`

Key artefacts: `k_recall_curves_five_scenarios.png`, `k_recall_points.csv`, `COMPARATIVE_ANALYSIS.md`, `anomaly_detection_comparative_table.tex`, `Why GRU Performs poor on Long GT Anomalies.md`.

GRU Δ-target ablation summaries: `results/gru_delta_ablation_restart/GRU_DELTA_ABLATION.md`, `k_recall_points.csv`.

## Repository layout

```
magnavis_v15/
├── README.md                 # Setup + entry points
├── requirements_feb_2025.txt
├── docs/                     # Architecture & script docs
├── Datafiles/                # CSVs (gitignored; README lists required files)
├── models/                   # Bundled pretrained LSTM/GRU (OBS1/OBS2)
├── src/
│   ├── app.py                # Primary DB/CSV runtime
│   ├── predictor_ai.py
│   ├── Anomaly_detector.py
│   ├── train_*_pretrained.py
│   └── benchmark_feb13_2026_improved/   # Main thesis benchmarks
├── tools/
└── thesis/                   # Optional: LaTeX includes / figure copies
```

## Recommended: new GitHub repo (not `magnavis_v4`)

The current `origin` still points at the older **magnavis_v4** project. For a clean thesis line, create a **new** repository (e.g. `magnavis_v15` or `magnavis-thesis`) and push there.

### One-time setup (after creating the empty repo on GitHub)

From the project root:

```bash
# 1) Inspect what Git would add (should be mostly code + docs + small models + result summaries)
git status
git add -n .   # dry-run

# 2) Point origin at the new repo (replace URL with yours)
git remote rename origin origin-v4-backup   # optional: keep old remote name
git remote add origin https://github.com/<user>/magnavis_v15.git

# 3) Stage and commit (only when you are ready)
git add .
git commit -m "Magnavis v15: thesis benchmark codebase and result summaries"

# 4) Push (use a fresh branch name if you want to avoid merging old v4 history)
git push -u origin main
```

### Fresh history (optional)

If you want **no** inherited commits from v4:

```bash
git checkout --orphan thesis-main
git add .
git commit -m "Initial commit: Magnavis v15 thesis codebase"
git branch -M main
git push -u origin main --force   # only on a NEW empty remote; never force main on shared repos
```

## Reproducing thesis figures

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_feb_2025.txt

# Copy required CSVs into Datafiles/ (see Datafiles/README.md)

# Regenerate k–recall comparative figure (after CSVs + models present)
python src/benchmark_feb13_2026_improved/generate_comparative_analysis.py
```

Full k–recall **re-run** (hours, needs TensorFlow + CSVs):

```bash
python src/benchmark_feb13_2026_improved/run_k_recall_curves.py
```

## Current working tree notes

Before the first thesis push, review `git status`:

- Many paths show as **deleted** relative to old v4 tracking (removed screenshots, shapefiles, old checkpoint names). That is expected when slimming the tree.
- Untracked `src/benchmark_feb13_2026_improved/` and updated `models/gru_pretrained_OBS2_*.keras` should be **added** for the thesis snapshot.
- Local folder size ~8+ GB is normal; the Git remote should stay **well under 100 MB** with the `.gitignore` rules in place.

## Checklist before push

- [ ] No secrets (DB passwords, API keys) in `src/` or `docs/`
- [ ] `Datafiles/*.csv` not staged (`git check-ignore -v Datafiles/*.csv`)
- [ ] No `src/sessions/` or `results/**/runs/` staged
- [ ] README and `docs/THESIS_REPOSITORY.md` up to date
- [ ] New GitHub repo created; `origin` URL updated

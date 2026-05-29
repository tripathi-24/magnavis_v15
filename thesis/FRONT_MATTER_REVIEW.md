# Front matter review (through Acknowledgements)

Review based on the IITK template copy synced into `thesis/`. Apply the same edits on Overleaf after Git pull.

## Blank pages — fixes applied

| Cause | Fix |
|-------|-----|
| `\cleardoublepage` in `Thesis.cls` (abstract, lists) | Added `\let\cleardoublepage\clearpage` at start of `main.tex` (oneside document) |
| `\vfil\vfil\null` after Certificate, Declaration, Acknowledgements | Replaced with modest `\vspace{2em}` in `Thesis.cls` |
| Extra `\clearpage` after abstract and acknowledgements | Removed from `main.tex` |
| Nearly empty dedication page | Dedication commented out in `main.tex` (uncomment if required) |
| Removed unused `\usepackage{lipsum}` | Dropped from `main.tex` |

Recompile and check the PDF page-by-page; one blank page between front-matter blocks is normal, but you should no longer see empty verso pages.

## Suggested improvements (content)

### Title page
- **Roll number:** still placeholder — add your IITK roll number in `main.tex` and in the abstract block (Roll No. in `Thesis.cls` abstract header).
- **Logo:** uncomment the IITK logo figure when `Pictures/iitk_logo.png` is available.

### Abstract (updated)
- Now reflects **v15** work: zero-historic benchmark, LSTM vs GRU on long GT, deployment note at $k \approx 3$.
- **Improvement:** add 2–3 keywords if your department requires them (e.g.\ geomagnetism, anomaly detection, LSTM).

### Acknowledgements (polished)
- More formal UK tone; names specific support without being generic.
- **Improvement:** add named lab colleagues or collaborators if you wish (with their permission).

### Certificate / Declaration
- Text is standard; ensure supervisor name matches official records.
- **Improvement:** attach scanned signature on declaration if required by Physics department format.

### Degree line
- Template says `Masters of Technology`; confirm with your programme (M.Tech vs MS vs other).

## What was not changed (you may want to)

- **Chapter 2** (literature review) — still the earlier draft; expand with citations from your reading list.
- **Chapter 6** (UI) — lightly aligned only; add screenshots from `app.py` if the department expects them.
- **Bibliography.bib** — populate with papers you actually cite in Chapter 2.

## Plagiarism note

New prose in abstract, Chapters 1, 5, and 7 is written from **your project artefacts** (benchmark CSVs, `COMPARATIVE_ANALYSIS.md`, code behaviour), not copied from external papers. Still run IITK’s checker on the final PDF and cite all external sources in Chapter 2.

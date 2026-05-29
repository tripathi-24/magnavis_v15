# Overleaf Git sync (Option A)

Link your Overleaf project to this repository so edits in `thesis/` sync both ways.

**Overleaf project:** https://www.overleaf.com/project/6a191ccc7918b3d8d436da6f  
**GitHub repo:** https://github.com/tripathi-24/magnavis_v15

## One-time setup in Overleaf

1. Open the project → **Menu** → **Git**.
2. Choose **GitHub** and authorise Overleaf.
3. Select repository **`tripathi-24/magnavis_v15`**.
4. Set the **Overleaf sub-path** to `thesis` (so only the thesis folder syncs, not the whole Python codebase).
5. Pull once to merge; resolve conflicts in favour of the version you trust if prompted.

## Workflow after sync

| Where you edit | Action |
|----------------|--------|
| Cursor (this folder) | Edit `thesis/*.tex` → `git add thesis` → `git commit` → `git push` → Overleaf **Pull** |
| Overleaf web | Edit online → Overleaf **Push** → `git pull` locally |

Compile PDF in Overleaf (recommended) or install MacTeX locally and run `pdflatex main.tex` from `thesis/`.

## Files updated in this pass

- `main.tex` — blank-page fixes, revised abstract and acknowledgements
- `Chapters/Chapter1.tex`, `Chapter5.tex`, `Chapter7.tex` — v15 benchmark content
- `Chapters/Chapter3.tex`, `Chapter4.tex` — minor alignment with current code
- `figures/` — k-recall plots for inclusion
- `tables/` — comparative results table (may need `\usepackage{multirow}` — already in `main.tex`)

## Before submission

1. Replace **Roll No.** placeholders in `main.tex` and `Thesis.cls` abstract header.
2. Add IITK logo to `Pictures/iitk_logo.png` if required.
3. Add signature image if your department requires it on the declaration page.
4. Run your institute plagiarism check on the compiled PDF.

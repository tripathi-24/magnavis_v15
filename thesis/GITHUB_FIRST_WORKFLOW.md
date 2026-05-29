# Thesis workflow: GitHub repo first (Overleaf second)

Overleaf **cannot** link project `6a191ccc...` to the existing repo `tripathi-24/magnavis_v15` in one step. Use **GitHub as the master copy** and Overleaf as a viewer/compiler.

---

## What you have

| Item | Location |
|------|----------|
| LaTeX thesis (updated) | https://github.com/tripathi-24/magnavis_v15/tree/main/thesis |
| Old Overleaf project | https://www.overleaf.com/project/6a191ccc7918b3d8d436da6f |
| Local clone | `magnavis_v15/thesis/` on your Mac |

---

## Recommended setup (one time)

### Step 1 — Save anything still only on Overleaf

In the **old** Overleaf project:

1. **Menu** (or project menu) → **Download** → **Source (zip)**.
2. Unzip on your Mac.
3. Copy any extra lines you need (roll number, logo, supervisor edits) into  
   `magnavis_v15/thesis/` locally.
4. Commit and push (see Step 2).

### Step 2 — Clone the repo locally (if not already)

```bash
cd ~/Documents/CSE@IITK/Summer_Term_26/Thesis_Work
git clone https://github.com/tripathi-24/magnavis_v15.git
cd magnavis_v15/thesis
```

Edit `.tex` files in Cursor. When done:

```bash
cd ..   # repo root magnavis_v15
git add thesis/
git commit -m "Thesis: describe your edit"
git push origin main
```

### Step 3 — New Overleaf project from GitHub

1. Go to https://www.overleaf.com/project
2. **New Project** → **Import from GitHub**  
   (If missing: use Step 4 “Upload zip” instead.)
3. Select repository **`magnavis_v15`**.
4. If asked for a path, choose **`thesis`**.
5. Open the new project → set **Main document** to **`main.tex`**.
6. **Recompile**.

Keep the old Overleaf project as backup; do new work in the **imported** project.

### Step 4 — If “Import from GitHub” is not available

1. Download thesis folder from GitHub:  
   https://github.com/tripathi-24/magnavis_v15/archive/refs/heads/main.zip  
2. Unzip → open inner folder **`magnavis_v15-main/thesis`**.
3. Zip the **`thesis`** folder contents (all `.tex`, `Thesis.cls`, `Chapters/`, etc.).
4. Overleaf → **New Project** → **Upload Project** → select that zip.
5. Main document: **`main.tex`**.

Later, when you have GitHub import or Integrations → GitHub on this **new** project, you can enable Push/Pull.

---

## Daily workflow (GitHub-centric)

```
┌─────────────┐     git push      ┌──────────────────┐
│ Cursor /    │ ────────────────► │ GitHub           │
│ local thesis│                   │ magnavis_v15/    │
└─────────────┘                   │ thesis/          │
       ▲                          └────────┬─────────┘
       │ git pull                           │
       │                          Pull (Integrations)
       │                          or re-import / upload
       │                                   ▼
       │                          ┌──────────────────┐
       └──────────────────────────│ Overleaf (new    │
              merge manual edits   │ project)         │
                                   └──────────────────┘
```

| Task | Where |
|------|--------|
| Write chapters, fix LaTeX | Cursor → `thesis/` → `git push` |
| PDF for supervisor | Overleaf → Recompile (after Pull if linked) |
| AI edits | Cursor on local clone, then push |

---

## Sync Overleaf ↔ GitHub (after import)

In the **new** imported project:

1. Left sidebar → **Integrations** → **GitHub**.
2. Link GitHub account if needed.
3. Use **Pull** (GitHub → Overleaf) after you `git push` from Mac.
4. Use **Push** (Overleaf → GitHub) only if you edited in Overleaf and want those changes in the repo.

If this project was created **from** GitHub, Pull/Push is more likely to work than on the old project.

---

## Optional: Git remote only on your Mac (no Overleaf GitHub UI)

Add GitHub as remote and push thesis changes:

```bash
cd magnavis_v15
git remote -v   # should show origin → magnavis_v15
git pull origin main
# edit thesis/
git add thesis && git commit -m "..." && git push origin main
```

Overleaf: re-download zip from GitHub occasionally, or use **Import from GitHub** once.

---

## Checklist

- [ ] Downloaded zip from old Overleaf; merged roll no. / logo into `thesis/`
- [ ] Pushed merged files to `magnavis_v15` on GitHub
- [ ] Created **new** Overleaf project (Import from GitHub or upload zip)
- [ ] Main file = `main.tex`; PDF compiles
- [ ] Future edits: Cursor → push → Overleaf Pull (or re-upload when needed)

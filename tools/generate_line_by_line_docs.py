#!/usr/bin/env python3
"""
Generate line-by-line documentation for selected Python scripts.

Outputs Markdown into ./docs/ explaining every single line (including blanks),
with additional higher-level structure summaries to make the docs usable.

This is intentionally deterministic and offline (no network calls).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
DOCS_DIR = REPO_ROOT / "docs"


TARGETS = [
    SRC_DIR / "application.py",
    SRC_DIR / "Anomaly_detector.py",
    SRC_DIR / "predictor_ai.py",
]


CHUNK_SIZE_BY_FILE = {
    "application.py": 200,  # keep human-navigable
    "Anomaly_detector.py": 999999,
    "predictor_ai.py": 999999,
}


KEYWORD_HINTS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bPyQt5\b|Q(Application|MainWindow|Widget|Label|Timer|Thread|Mutex|FileDialog)\b"), "PyQt5/Qt GUI framework usage"),
    (re.compile(r"\bvtk\b|vtk[A-Z]\w+"), "VTK (3D visualization) usage"),
    (re.compile(r"\bmatplotlib\b|FigureCanvas|NavigationToolbar"), "Matplotlib plotting (Qt backend) usage"),
    (re.compile(r"\bGeoMag\b|\bpygeomag\b"), "World Magnetic Model (geomagnetic field) usage"),
    (re.compile(r"\bgeopandas\b|\bgpd\b|\bshapely\b"), "Geospatial libraries (GeoPandas/Shapely) usage"),
    (re.compile(r"\bsubprocess\.Popen\b"), "Spawns a subprocess (external predictor process)"),
    (re.compile(r"\bAnomalyDetector\b"), "Anomaly detection integration"),
    (re.compile(r"\bLSTM\b|\btensorflow\b|\bkeras\b"), "LSTM / TensorFlow model code"),
    (re.compile(r"\bMinMaxScaler\b"), "Feature scaling for ML (sklearn MinMaxScaler)"),
    (re.compile(r"\bmplcursors\b"), "Interactive plot tooltips via mplcursors"),
]


def _escape_md_code(s: str) -> str:
    """Escape content for use inside a Markdown inline-code cell."""
    s = s.rstrip("\n")
    s = s.replace("\\", "\\\\")
    s = s.replace("|", "\\|")
    # Inline code is wrapped in single backticks; escape backticks by using HTML entity.
    s = s.replace("`", "&#96;")
    return s


@dataclass
class Scope:
    kind: str  # "module" | "class" | "def"
    name: str
    indent: int


def _indent_level(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _parse_def_name(stripped: str) -> Optional[str]:
    m = re.match(r"def\s+([A-Za-z_]\w*)\s*\(", stripped)
    return m.group(1) if m else None


def _parse_class_name(stripped: str) -> Optional[str]:
    m = re.match(r"class\s+([A-Za-z_]\w*)\s*(\(|:)", stripped)
    return m.group(1) if m else None


def _hint_for_line(stripped: str) -> Optional[str]:
    for pat, hint in KEYWORD_HINTS:
        if pat.search(stripped):
            return hint
    return None


def _top_level_assignment_operator(stripped: str) -> Optional[Tuple[str, str]]:
    """
    Detect a top-level assignment operator in a line.

    Returns (op, lhs) where:
    - op is one of "=", "+=", "-=", "*=", "/=", "//=", "%=", "**=", "&=", "|=", "^=", ">>=", "<<=", ":="
    - lhs is the left-hand side text (best-effort).

    Important: ignores '=' that occur inside parentheses/brackets/braces (e.g., keyword arguments),
    and ignores comparison operators like '==', '!=', '<=', '>='.
    """
    # Quick exits for comment-only lines
    if not stripped or stripped.startswith("#"):
        return None

    depth_paren = 0
    depth_brack = 0
    depth_brace = 0
    in_single = False
    in_double = False
    escaped = False

    i = 0
    while i < len(stripped):
        ch = stripped[i]

        if escaped:
            escaped = False
            i += 1
            continue

        if ch == "\\":
            escaped = True
            i += 1
            continue

        if not in_double and ch == "'" and (i == 0 or stripped[i - 1] != "\\"):
            in_single = not in_single
            i += 1
            continue
        if not in_single and ch == '"' and (i == 0 or stripped[i - 1] != "\\"):
            in_double = not in_double
            i += 1
            continue

        if in_single or in_double:
            i += 1
            continue

        if ch == "(":
            depth_paren += 1
            i += 1
            continue
        if ch == ")":
            depth_paren = max(0, depth_paren - 1)
            i += 1
            continue
        if ch == "[":
            depth_brack += 1
            i += 1
            continue
        if ch == "]":
            depth_brack = max(0, depth_brack - 1)
            i += 1
            continue
        if ch == "{":
            depth_brace += 1
            i += 1
            continue
        if ch == "}":
            depth_brace = max(0, depth_brace - 1)
            i += 1
            continue

        # Only consider operators at top level (not inside (), [], {})
        if depth_paren == 0 and depth_brack == 0 and depth_brace == 0:
            # Check multi-char operators first
            for op in ("<<=", ">>=", "//=", "**=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", ":="):
                if stripped.startswith(op, i):
                    lhs = stripped[:i].strip()
                    return op, lhs

            # Plain '=': ensure not part of '==', '!=', '<=', '>='
            if ch == "=":
                prev = stripped[i - 1] if i - 1 >= 0 else ""
                nxt = stripped[i + 1] if i + 1 < len(stripped) else ""
                if nxt == "=":
                    i += 1
                    continue
                if prev in {"!", "<", ">", "="}:
                    i += 1
                    continue
                lhs = stripped[:i].strip()
                return "=", lhs

        i += 1

    return None


def _explain_line(
    *,
    line: str,
    line_no: int,
    scopes: List[Scope],
    prev_line: str,
    next_line: str,
) -> str:
    """
    Heuristic, context-aware explanation of a single line.

    Goal: every line gets a meaningful explanation, even blank lines.
    """
    raw = line.rstrip("\n")
    stripped = raw.strip()

    scope_str = ""
    if scopes:
        scope_path = " → ".join([f"{s.kind} {s.name}" for s in scopes])
        scope_str = f" (scope: {scope_path})"

    if stripped == "":
        return "Blank line used to separate logical sections and improve readability." + scope_str

    if stripped.startswith("#"):
        comment = stripped[1:].strip()
        if comment == "":
            return "Comment line used as a visual separator." + scope_str
        return f"Comment explaining intent/context: {comment}" + scope_str

    # Docstring / multiline string markers (common at module top or inside defs/classes)
    if stripped in {"'''", '"""'}:
        # Try to infer begin vs end by checking neighborhood.
        # Not perfect, but still documents the line.
        if prev_line.strip() == "" or prev_line.strip().startswith("#"):
            return "Begins a triple-quoted string (often used as a module/class/function docstring)." + scope_str
        return "Ends a triple-quoted string (closing a docstring/multiline literal)." + scope_str

    if stripped.startswith(("'''", '"""')) and stripped.endswith(("'''", '"""')) and len(stripped) > 6:
        return "Single-line triple-quoted string (docstring or multiline literal expressed on one line)." + scope_str

    if stripped.startswith("import "):
        mods = stripped[len("import ") :].strip()
        return f"Imports module(s) into this namespace: `{mods}`." + scope_str

    if stripped.startswith("from "):
        return f"Imports specific symbols using a `from … import …` statement: `{stripped}`." + scope_str

    if stripped.startswith("class "):
        name = _parse_class_name(stripped) or "<unknown>"
        return f"Defines class `{name}` (starts a new type/namespace for related behavior)." + scope_str

    if stripped.startswith("def "):
        name = _parse_def_name(stripped) or "<unknown>"
        return f"Defines function/method `{name}` (entry point for reusable logic)." + scope_str

    if stripped.endswith(":"):
        head = stripped[:-1].split()[0] if stripped[:-1].split() else ""
        if head == "if":
            return f"Starts an `if` block: conditional control flow based on `{stripped[3:-1].strip()}`." + scope_str
        if head == "elif":
            return f"Starts an `elif` branch: additional conditional `{stripped[5:-1].strip()}`." + scope_str
        if stripped == "else:":
            return "Starts an `else` branch for the preceding conditional." + scope_str
        if head == "for":
            return f"Starts a `for` loop: iterates as described by `{stripped[4:-1].strip()}`." + scope_str
        if head == "while":
            return f"Starts a `while` loop controlled by `{stripped[6:-1].strip()}`." + scope_str
        if head == "try":
            return "Starts a `try` block to catch and handle exceptions." + scope_str
        if head == "except":
            return f"Starts an `except` handler: `{stripped}`." + scope_str
        if head == "finally":
            return "Starts a `finally` block: runs cleanup regardless of whether an exception occurred." + scope_str
        if head == "with":
            return f"Starts a `with` block (context manager) for `{stripped[5:-1].strip()}`." + scope_str

    if stripped.startswith("return"):
        expr = stripped[len("return") :].strip()
        if expr:
            return f"Returns value(s) from the current function: `{expr}`." + scope_str
        return "Returns from the current function (no explicit value, so `None`)." + scope_str

    if stripped.startswith(("raise ", "raise")):
        return f"Raises an exception to signal an error condition: `{stripped}`." + scope_str

    if stripped in {"pass", "continue", "break"}:
        if stripped == "pass":
            return "No-op placeholder statement (keeps block syntactically valid)." + scope_str
        if stripped == "continue":
            return "Skips to the next iteration of the current loop." + scope_str
        return "Exits the current loop immediately." + scope_str

    # Common patterns for this repo
    hint = _hint_for_line(stripped)
    if ".connect(" in stripped:
        return "Connects a Qt signal to a slot/callback so UI events trigger Python code." + (f" Hint: {hint}." if hint else "") + scope_str
    if "pyqtSignal" in stripped:
        return "Declares a Qt signal (used for thread-safe event notifications between objects/threads)." + scope_str
    if "QTimer" in stripped and ("singleShot" in stripped or "new_timer" in stripped or ".start" in stripped):
        return "Configures/starts a timer to periodically trigger updates without blocking the UI thread." + scope_str
    if "subprocess.Popen" in stripped:
        return "Starts an external process (here: the predictor script) and captures its output for later use." + scope_str

    # Assignment / augmented assignment (top-level only; ignores keyword-arg '=' inside calls)
    assign = _top_level_assignment_operator(stripped)
    if assign is not None:
        op, lhs = assign
        if op == "=":
            return f"Assigns/updates `{lhs}` with the expression on the right-hand side." + (f" Hint: {hint}." if hint else "") + scope_str
        if op == ":=":
            return f"Uses the walrus operator `:=` to assign-and-return a value into `{lhs}` within an expression." + (f" Hint: {hint}." if hint else "") + scope_str
        return f"Updates `{lhs}` using augmented assignment `{op}` (reads current value, applies operation, stores back)." + (f" Hint: {hint}." if hint else "") + scope_str

    # Function/method calls, attribute access, etc.
    if hint:
        return f"Executes statement related to: {hint}: `{stripped}`." + scope_str

    return f"Executes Python statement: `{stripped}`." + scope_str


def _extract_structure(lines: List[str]) -> List[str]:
    """Lightweight structure summary: list defs/classes with approximate line numbers."""
    items: List[str] = []
    scopes: List[Scope] = []
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        indent = _indent_level(line)

        # Pop scopes on dedent
        while scopes and indent < scopes[-1].indent:
            scopes.pop()

        if stripped.startswith("class "):
            name = _parse_class_name(stripped)
            if name:
                items.append(f"- Line {i}: class `{name}`")
                scopes.append(Scope(kind="class", name=name, indent=indent + 4))
        elif stripped.startswith("def "):
            name = _parse_def_name(stripped)
            if name:
                prefix = "method" if scopes and scopes[-1].kind == "class" else "function"
                items.append(f"- Line {i}: {prefix} `{name}()`")
                scopes.append(Scope(kind="def", name=name, indent=indent + 4))
    return items


def _write_markdown_for_file(path: Path) -> List[Path]:
    rel = path.relative_to(REPO_ROOT)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    total = len(lines)

    chunk_size = CHUNK_SIZE_BY_FILE.get(path.name, 400)
    out_paths: List[Path] = []

    # Output directory per file (only when chunked)
    if total > chunk_size:
        out_dir = DOCS_DIR / f"{path.name}_line_by_line"
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = DOCS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

    structure = _extract_structure(lines)

    def write_chunk(start_idx: int, end_idx: int, out_path: Path) -> None:
        scopes: List[Scope] = []
        md: List[str] = []

        md.append(f"# Line-by-line documentation: `{rel.as_posix()}`")
        md.append("")
        md.append(f"- **File**: `{rel.as_posix()}`")
        md.append(f"- **Lines covered in this document**: {start_idx}–{end_idx} (of {total})")
        md.append("")
        md.append("## Structure overview (file-level)")
        md.append("")
        if structure:
            md.extend(structure)
        else:
            md.append("- (No `class`/`def` blocks detected.)")
        md.append("")
        md.append("## Line-by-line explanation")
        md.append("")
        md.append("| Line | Code | Explanation |")
        md.append("|---:|---|---|")

        for ln in range(start_idx, end_idx + 1):
            line = lines[ln - 1]
            prev_line = lines[ln - 2] if ln - 2 >= 0 else ""
            next_line = lines[ln] if ln < total else ""
            indent = _indent_level(line)
            stripped = line.strip()

            # maintain scopes
            while scopes and indent < scopes[-1].indent:
                scopes.pop()

            if stripped.startswith("class "):
                cname = _parse_class_name(stripped) or "<unknown>"
                scopes.append(Scope(kind="class", name=cname, indent=indent + 4))
            elif stripped.startswith("def "):
                fname = _parse_def_name(stripped) or "<unknown>"
                scopes.append(Scope(kind="def", name=fname, indent=indent + 4))

            explanation = _explain_line(
                line=line,
                line_no=ln,
                scopes=scopes,
                prev_line=prev_line,
                next_line=next_line,
            )
            code_cell = _escape_md_code(line)
            md.append(f"| {ln} | `{code_cell}` | {explanation} |")

        out_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    if total <= chunk_size:
        out_path = out_dir / f"{path.name}.md"
        write_chunk(1, total, out_path)
        out_paths.append(out_path)
    else:
        part = 1
        for start in range(1, total + 1, chunk_size):
            end = min(total, start + chunk_size - 1)
            out_path = out_dir / f"part_{part:03d}_lines_{start:04d}_{end:04d}.md"
            write_chunk(start, end, out_path)
            out_paths.append(out_path)
            part += 1

        # create index for chunked output
        index_path = out_dir / "INDEX.md"
        rel_out_dir = out_dir.relative_to(REPO_ROOT)
        links = [f"- [{p.name}]({p.name})" for p in out_paths]
        index_md = [
            f"# Index: `{rel.as_posix()}` line-by-line docs",
            "",
            f"This directory contains chunked line-by-line documentation for `{rel.as_posix()}`.",
            "",
            f"- **Chunk size**: {chunk_size} lines",
            f"- **Total lines**: {total}",
            "",
            "## Parts",
            "",
            *links,
            "",
        ]
        index_path.write_text("\n".join(index_md), encoding="utf-8")
        out_paths.append(index_path)

    return out_paths


def _write_docs_index(generated: List[Path]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    # Map outputs for navigation
    by_target = {}
    for p in generated:
        try:
            rel = p.relative_to(DOCS_DIR)
        except Exception:
            continue
        # bucket by top-level component
        top = rel.parts[0] if rel.parts else rel.as_posix()
        by_target.setdefault(top, []).append(rel)

    lines = [
        "# Documentation (auto-generated)",
        "",
        "These documents were regenerated from scratch and provide:",
        "",
        "- A file-level structure overview (`class`/`def` inventory).",
        "- A **line-by-line explanation of every single line** in:",
        "  - `src/application.py`",
        "  - `src/Anomaly_detector.py`",
        "  - `src/predictor_ai.py`",
        "",
        "## Entry points",
        "",
        "- `application.py` (chunked): `application.py_line_by_line/INDEX.md`",
        "- `Anomaly_detector.py`: `Anomaly_detector.py.md`",
        "- `predictor_ai.py`: `predictor_ai.py.md`",
        "",
    ]
    (DOCS_DIR / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
    for target in TARGETS:
        if not target.exists():
            raise FileNotFoundError(f"Missing target: {target}")
        generated.extend(_write_markdown_for_file(target))
    _write_docs_index(generated)
    print(f"Generated {len(generated)} markdown files under: {DOCS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



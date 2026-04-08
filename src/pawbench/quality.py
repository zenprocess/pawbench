"""Artifact quality analyzers — spec 009 / B4.

Static-analysis scoring over the *artifact* an agent produced (the code
extracted from tool calls), orthogonal to AC pass/fail. Catches the
"passes tests, ships slop" failure mode.

This is intentionally **not** folded into composite quality scores yet —
calibration data first (≥100 dispatches), formula change later.

Analyzers are pluggable: every language registers a callable that takes
extracted source files and returns an `ArtifactQuality`. Missing tools
degrade gracefully (analyzer returns a `None` score with `analyzer=""`).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass
class ArtifactQuality:
    """Static-analysis score over changed files (Axiom §17.5).

    `score` is normalized 0..1 where 1.0 = clean. Analyzers MUST clamp.
    `lint_errors`, `type_errors`, `cyclomatic_max` are raw counts.
    `analyzer` identifies the toolchain ("ruff+mypy+radon", "eslint+tsc"...).
    Empty `analyzer` means the analyzer was unavailable; consumers should
    treat this row as "no signal" rather than "perfect score".
    """

    language: str
    lint_errors: int = 0
    type_errors: int = 0
    cyclomatic_max: int = 0
    score: float = 0.0
    analyzer: str = ""
    files_analyzed: int = 0
    notes: str = ""

    @property
    def is_signal(self) -> bool:
        """True if this row reflects an actual analyzer run."""
        return bool(self.analyzer)


# ---------------------------------------------------------------------------
# File extraction from tool calls
# ---------------------------------------------------------------------------


def extract_files_from_tool_calls(
    tool_calls: Iterable[dict[str, Any]],
) -> dict[str, str]:
    """Extract `path -> content` from `write_file` tool calls.

    Pawbench scenarios use `write_file(path, content)` as the canonical
    artifact-emission tool. Other tools are ignored. Duplicate paths
    keep the last write (mirrors filesystem semantics).
    """
    files: dict[str, str] = {}
    for tc in tool_calls:
        fn = tc.get("function", {})
        if fn.get("name") != "write_file":
            continue
        raw_args = fn.get("arguments", "")
        if not raw_args:
            continue
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            continue
        path = args.get("path")
        content = args.get("content")
        if isinstance(path, str) and isinstance(content, str):
            files[path] = content
    return files


def detect_language(files: dict[str, str]) -> str:
    """Pick the dominant language by file extension."""
    if not files:
        return "unknown"
    counts: dict[str, int] = {}
    for path in files:
        suffix = Path(path).suffix.lower()
        lang = _SUFFIX_TO_LANG.get(suffix)
        if lang:
            counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return "unknown"
    return max(counts.items(), key=lambda kv: kv[1])[0]


_SUFFIX_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
    ".html": "html",
    ".css": "css",
    ".sh": "shell",
}


# ---------------------------------------------------------------------------
# Python analyzer (ruff + mypy + radon, all optional)
# ---------------------------------------------------------------------------


def _which(name: str) -> str | None:
    return shutil.which(name)


def _run(cmd: list[str], cwd: Path, timeout: int = 60) -> tuple[int, str, str]:
    """Run a static-analyzer subprocess.

    Security review (spec 009 / B4):
      - shell=False (default) — no shell metacharacter expansion
      - cmd is a list of explicit arguments built from `shutil.which()` paths
        plus a single trusted scratch directory path; no user-controlled
        token reaches argv
      - cwd is the temporary scratch dir created by tempfile.TemporaryDirectory
      - timeout is bounded; OSError/TimeoutExpired are caught
      - capture_output=True — no stdio leak to the parent process
    The user-controlled artifact bytes never reach this function as argv;
    they're written to disk first by _materialize and addressed by file path.
    """
    if not cmd or not isinstance(cmd[0], str):
        return -1, "", "invalid command"
    try:
        proc = subprocess.run(  # nosec B603 — argv-list, fixed scratch cwd, no shell
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            shell=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return -1, "", str(e)


def _materialize(files: dict[str, str], root: Path) -> list[Path]:
    """Write files to disk under root, return absolute paths.

    Security review: every relative path is resolved and re-anchored under
    `root`; any path that escapes the scratch dir (via `..`, absolute path,
    or symlink trick) is silently rejected. Content bytes are written
    verbatim because the analyzers expect them as source files; no
    interpretation happens here.
    """
    written: list[Path] = []
    root_resolved = root.resolve()
    for rel, content in files.items():
        target = (root / rel).resolve()
        try:
            target.relative_to(root_resolved)
        except ValueError:
            continue  # path escape attempt — silently drop
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(target)
    return written


def _count_ruff_issues(out: str) -> int:
    try:
        issues = json.loads(out) if out.strip() else []
        return len(issues) if isinstance(issues, list) else 0
    except json.JSONDecodeError:
        return out.count('"code":')


def _count_mypy_errors(out: str) -> int:
    return sum(1 for line in out.splitlines() if ": error:" in line)


def _max_radon_complexity(out: str) -> int:
    if not out.strip():
        return 0
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return 0
    max_cc = 0
    for entries in data.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            cc = entry.get("complexity", 0)
            if isinstance(cc, (int, float)) and cc > max_cc:
                max_cc = int(cc)
    return max_cc


def _run_python_analyzers(tools: dict[str, str | None], root: Path, aq: ArtifactQuality) -> None:
    """Mutate `aq` with results from each available analyzer."""
    if tools["ruff"]:
        rc, out, _ = _run(
            [tools["ruff"], "check", "--output-format=json", "--exit-zero", str(root)],
            cwd=root,
        )
        if rc >= 0:
            aq.lint_errors = _count_ruff_issues(out)

    if tools["mypy"]:
        rc, out, _ = _run(
            [tools["mypy"], "--ignore-missing-imports", "--no-error-summary", "--no-color-output", str(root)],
            cwd=root,
            timeout=90,
        )
        if rc >= 0:
            aq.type_errors = _count_mypy_errors(out)

    if tools["radon"]:
        rc, out, _ = _run([tools["radon"], "cc", "-j", "-s", str(root)], cwd=root)
        if rc >= 0:
            aq.cyclomatic_max = _max_radon_complexity(out)


def _analyze_python(files: dict[str, str]) -> ArtifactQuality:
    py_files = {p: c for p, c in files.items() if p.endswith(".py")}
    if not py_files:
        return ArtifactQuality(language="python", notes="no python files")

    tools: dict[str, str | None] = {
        "ruff": _which("ruff"),
        "mypy": _which("mypy"),
        "radon": _which("radon"),
    }
    available = [name for name, path in tools.items() if path]
    if not available:
        return ArtifactQuality(
            language="python",
            files_analyzed=len(py_files),
            notes="no analyzers available (install ruff/mypy/radon)",
        )

    aq = ArtifactQuality(
        language="python",
        files_analyzed=len(py_files),
        analyzer="+".join(available),
    )

    with tempfile.TemporaryDirectory(prefix="pawbench-quality-") as td:
        root = Path(td)
        if not _materialize(py_files, root):
            aq.notes = "all paths rejected (escape attempts)"
            aq.analyzer = ""
            return aq
        _run_python_analyzers(tools, root, aq)

    aq.score = _score_python(aq)
    return aq


def _score_python(aq: ArtifactQuality) -> float:
    """Bounded 0..1 quality score from raw counts.

    Cheap, transparent, and explicitly NOT a learned model. We start with
    1.0 and subtract bounded penalties so the formula is auditable. The
    weights are tuned to be lenient on small artifacts (Pawbench scenarios
    are typically <500 LOC) and to penalize hot spots more than dispersion.
    """
    if not aq.is_signal or aq.files_analyzed == 0:
        return 0.0

    files = max(aq.files_analyzed, 1)
    # Lint errors per file, capped at 10/file => 0.4 max penalty
    lint_density = min(aq.lint_errors / files, 10) / 10 * 0.4
    # Type errors per file, capped at 5/file => 0.4 max penalty
    type_density = min(aq.type_errors / files, 5) / 5 * 0.4
    # Cyclomatic complexity hot spot above 10 => up to 0.2 penalty at 30+
    cc_penalty = max(0, min(aq.cyclomatic_max - 10, 20)) / 20 * 0.2

    score = 1.0 - lint_density - type_density - cc_penalty
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Generic / fallback analyzer
# ---------------------------------------------------------------------------

_FILLER_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")


def _analyze_generic(files: dict[str, str], language: str) -> ArtifactQuality:
    """Tool-free fallback: count obvious smell signals.

    Used when no language-specific analyzer is registered. Produces a
    real signal (`analyzer="generic"`) so downstream consumers can still
    differentiate empty artifacts from analyzed ones.
    """
    if not files:
        return ArtifactQuality(language=language, notes="no files")

    total_lines = 0
    smell_hits = 0
    longest_function = 0
    for content in files.values():
        lines = content.splitlines()
        total_lines += len(lines)
        smell_hits += len(_FILLER_RE.findall(content))
        # Cheap proxy for "long function": longest run of indented lines
        run = 0
        for line in lines:
            if line.startswith((" ", "\t")):
                run += 1
                longest_function = max(longest_function, run)
            else:
                run = 0

    aq = ArtifactQuality(
        language=language,
        files_analyzed=len(files),
        analyzer="generic",
        lint_errors=smell_hits,
        cyclomatic_max=longest_function,
    )
    # Score: penalize smell density and very long functions.
    smell_density = min(smell_hits / max(total_lines, 1) * 100, 5) / 5 * 0.5
    length_penalty = max(0, min(longest_function - 50, 50)) / 50 * 0.3
    aq.score = max(0.0, 1.0 - smell_density - length_penalty)
    return aq


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

Analyzer = Callable[[dict[str, str]], ArtifactQuality]

_REGISTRY: dict[str, Analyzer] = {
    "python": _analyze_python,
}


def register_analyzer(language: str, analyzer: Analyzer) -> None:
    """Register a per-language analyzer. Tests use this to inject fakes."""
    _REGISTRY[language] = analyzer


def analyze_artifact(tool_calls: Iterable[dict[str, Any]]) -> ArtifactQuality:
    """Top-level entry point: extract → detect language → analyze.

    Always returns an `ArtifactQuality`, never None. If nothing was
    written, returns an empty signal-less row.
    """
    files = extract_files_from_tool_calls(tool_calls)
    if not files:
        return ArtifactQuality(language="unknown", notes="no write_file calls")
    language = detect_language(files)
    analyzer = _REGISTRY.get(language)
    if analyzer is not None:
        return analyzer(files)
    return _analyze_generic(files, language)

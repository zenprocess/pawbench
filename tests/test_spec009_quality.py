"""Spec 009 / B4 — artifact quality analyzer tests."""

from __future__ import annotations

import json

from pawbench.quality import (
    ArtifactQuality,
    _analyze_generic,
    _score_python,
    analyze_artifact,
    detect_language,
    extract_files_from_tool_calls,
    register_analyzer,
)


def _wf(path: str, content: str) -> dict:
    return {
        "function": {
            "name": "write_file",
            "arguments": json.dumps({"path": path, "content": content}),
        }
    }


def test_extract_files_from_write_file_calls():
    calls = [_wf("a.py", "print(1)"), _wf("b.py", "print(2)")]
    files = extract_files_from_tool_calls(calls)
    assert files == {"a.py": "print(1)", "b.py": "print(2)"}


def test_extract_ignores_non_write_tools():
    calls = [
        _wf("a.py", "x"),
        {"function": {"name": "read_file", "arguments": '{"path":"a.py"}'}},
    ]
    assert list(extract_files_from_tool_calls(calls)) == ["a.py"]


def test_extract_handles_invalid_json():
    calls = [{"function": {"name": "write_file", "arguments": "not json"}}]
    assert extract_files_from_tool_calls(calls) == {}


def test_extract_dedupes_paths_keeping_last():
    files = extract_files_from_tool_calls([_wf("a.py", "v1"), _wf("a.py", "v2")])
    assert files == {"a.py": "v2"}


def test_detect_language_dominant_extension():
    assert detect_language({"a.py": "", "b.py": "", "c.go": ""}) == "python"
    assert detect_language({"a.ts": "", "b.tsx": ""}) == "typescript"
    assert detect_language({}) == "unknown"
    assert detect_language({"a.weird": ""}) == "unknown"


def test_score_python_clean_artifact_is_high():
    aq = ArtifactQuality(
        language="python",
        lint_errors=0,
        type_errors=0,
        cyclomatic_max=5,
        files_analyzed=3,
        analyzer="ruff+mypy+radon",
    )
    assert _score_python(aq) == 1.0


def test_score_python_no_signal_returns_zero():
    aq = ArtifactQuality(language="python", files_analyzed=0)
    assert _score_python(aq) == 0.0


def test_score_python_penalizes_lint_density():
    aq = ArtifactQuality(
        language="python",
        lint_errors=20,
        type_errors=0,
        cyclomatic_max=5,
        files_analyzed=2,
        analyzer="ruff",
    )
    # 20/2 = 10 errors/file → max lint penalty (0.4)
    assert _score_python(aq) == 0.6


def test_score_python_clamps_below_zero():
    aq = ArtifactQuality(
        language="python",
        lint_errors=200,
        type_errors=200,
        cyclomatic_max=100,
        files_analyzed=1,
        analyzer="ruff+mypy+radon",
    )
    assert _score_python(aq) == 0.0


def test_generic_analyzer_flags_smell_keywords():
    aq = _analyze_generic({"a.go": "// TODO: rewrite\nfunc x() {}\n"}, "go")
    assert aq.analyzer == "generic"
    assert aq.lint_errors == 1
    assert 0.0 <= aq.score <= 1.0


def test_analyze_artifact_empty_returns_signal_less_row():
    aq = analyze_artifact([])
    assert aq.language == "unknown"
    assert aq.is_signal is False
    assert aq.score == 0.0


def test_analyze_artifact_dispatches_to_registered_analyzer():
    seen: dict = {}

    def fake(files):
        seen["files"] = files
        return ArtifactQuality(language="ruby", files_analyzed=len(files), analyzer="fake", score=0.77)

    register_analyzer("ruby", fake)
    aq = analyze_artifact([_wf("a.rb", "puts 1")])
    assert aq.analyzer == "fake"
    assert aq.score == 0.77
    assert seen["files"] == {"a.rb": "puts 1"}


def test_analyze_artifact_falls_back_to_generic_for_unknown_language():
    aq = analyze_artifact([_wf("a.go", "package main\nfunc main() {}\n")])
    assert aq.language == "go"
    assert aq.analyzer == "generic"
    assert aq.is_signal is True

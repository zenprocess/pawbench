"""Tests for pawbench.leaderboard module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pawbench.leaderboard import render_leaderboard, submit, validate_submission

VALID_SUBMISSION = {
    "model": "Qwen3-Coder-80B-FP8",
    "gpu": "NVIDIA GB10 (128GB UMA)",
    "config": {
        "serving_engine": "vLLM",
        "quantization": "FP8",
        "tensor_parallel": 1,
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.78,
        "enforce_eager": True,
    },
    "single_tok_s": 18.4,
    "peak_tok_s": 52.7,
    "peak_concurrency": 4,
    "avg_quality": 0.82,
    "avg_ttft_ms": 342.1,
    "format_compliance_rate": 0.91,
    "tool_accuracy": 0.76,
    "useful_ratio": 0.68,
    "steering_rate": 0.85,
    "scenarios_run": 4,
    "runs_per_scenario": 2,
    "concurrency_levels": [1, 2, 4, 8],
    "timestamp": "2026-03-25T14:30:00Z",
    "pawbench_version": "0.1.0",
    "tag": "gb10-qwen3-fp8-baseline",
    "notes": "Eagle3-spec3 speculative decoding enabled.",
}


@pytest.fixture
def valid_file(tmp_path: Path) -> Path:
    p = tmp_path / "valid.json"
    p.write_text(json.dumps(VALID_SUBMISSION))
    return p


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    return d


class TestValidateSubmission:
    def test_valid_submission(self, valid_file: Path) -> None:
        errors = validate_submission(valid_file)
        assert errors == []

    def test_missing_required_field(self, tmp_path: Path) -> None:
        data = {k: v for k, v in VALID_SUBMISSION.items() if k != "model"}
        p = tmp_path / "missing_model.json"
        p.write_text(json.dumps(data))
        errors = validate_submission(p)
        assert any("model" in e for e in errors)

    def test_invalid_quality_range(self, tmp_path: Path) -> None:
        data = {**VALID_SUBMISSION, "avg_quality": 1.5}
        p = tmp_path / "bad_quality.json"
        p.write_text(json.dumps(data))
        errors = validate_submission(p)
        assert any("avg_quality" in e for e in errors)

    def test_wrong_type(self, tmp_path: Path) -> None:
        data = {**VALID_SUBMISSION, "single_tok_s": "fast"}
        p = tmp_path / "bad_type.json"
        p.write_text(json.dumps(data))
        errors = validate_submission(p)
        assert any("single_tok_s" in e for e in errors)

    def test_unknown_field_rejected(self, tmp_path: Path) -> None:
        data = {**VALID_SUBMISSION, "secret_sauce": "trust me"}
        p = tmp_path / "extra_field.json"
        p.write_text(json.dumps(data))
        errors = validate_submission(p)
        assert any("secret_sauce" in e for e in errors)

    def test_file_not_found(self) -> None:
        errors = validate_submission("/nonexistent/path.json")
        assert any("not found" in e.lower() for e in errors)

    def test_invalid_json(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        errors = validate_submission(p)
        assert any("Invalid JSON" in e for e in errors)

    def test_missing_nested_required(self, tmp_path: Path) -> None:
        data = {**VALID_SUBMISSION, "config": {"quantization": "FP8"}}
        p = tmp_path / "bad_config.json"
        p.write_text(json.dumps(data))
        errors = validate_submission(p)
        assert any("serving_engine" in e for e in errors)


class TestRenderLeaderboard:
    def test_render_with_submissions(self, results_dir: Path) -> None:
        for i, tok in enumerate([52.7, 80.1, 30.0]):
            data = {**VALID_SUBMISSION, "peak_tok_s": tok, "model": f"Model-{i}"}
            (results_dir / f"result_{i}.json").write_text(json.dumps(data))

        table = render_leaderboard(results_dir)
        assert "PawBench Leaderboard" in table
        assert "Model-1" in table
        # Model-1 (80.1) should be rank 1
        lines = table.split("\n")
        data_lines = [row for row in lines if row.startswith("| ") and "Rank" not in row and "---" not in row]
        assert "Model-1" in data_lines[0]
        assert "3 submission(s)" in table

    def test_render_empty_dir(self, results_dir: Path) -> None:
        table = render_leaderboard(results_dir)
        assert "No valid submissions" in table

    def test_render_missing_dir(self, tmp_path: Path) -> None:
        table = render_leaderboard(tmp_path / "nonexistent")
        assert "No results directory" in table


class TestSubmit:
    def test_submit_valid(self, valid_file: Path, results_dir: Path) -> None:
        errors = submit(valid_file, results_dir)
        assert errors == []
        assert (results_dir / valid_file.name).exists()

    def test_submit_invalid_blocked(self, tmp_path: Path, results_dir: Path) -> None:
        data = {**VALID_SUBMISSION}
        del data["model"]
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(data))
        errors = submit(p, results_dir)
        assert len(errors) > 0
        assert not (results_dir / p.name).exists()

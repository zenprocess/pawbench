"""Tests for pawbench.cli module."""

from __future__ import annotations

import subprocess
import sys


class TestCliHelp:
    def test_help_flag(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "pawbench.cli", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        assert "PawBench" in result.stdout
        assert "--endpoint" in result.stdout
        assert "--concurrency" in result.stdout

    def test_version_flag(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "pawbench.cli", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0
        # Version string should be present in output
        assert "0.1.0" in result.stdout or "pawbench" in result.stdout


class TestCompareHelp:
    def test_compare_no_args_exits(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "pawbench.compare"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 1
        assert "Usage" in result.stdout

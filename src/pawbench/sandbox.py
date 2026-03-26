"""Execution-based correctness scoring via subprocess sandbox.

Extracts files from agent tool_call arguments, writes them to a temp
directory, starts the server, hits expected endpoints, and scores
the result as endpoints_passing / endpoints_total.

No Docker dependency -- uses subprocess + tempdir (v1.1).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from pawbench.endpoint_spec import (
    EndpointSpec,
    ScenarioEndpoints,
    get_endpoint_specs,
    validate_response,
)
from pawbench.types import AgentResult

logger = logging.getLogger(__name__)

SERVER_START_TIMEOUT = 10  # seconds to wait for server to become ready
SERVER_RUN_TIMEOUT = 10  # seconds to allow the server process to live
REQUEST_TIMEOUT = 5  # per-request timeout


@dataclass
class EndpointResult:
    """Result of testing one endpoint."""

    spec: EndpointSpec
    passed: bool = False
    status_code: int | None = None
    error: str = ""


@dataclass
class SandboxResult:
    """Full sandbox evaluation result for one agent."""

    agent_id: str
    score: float = 0.0
    endpoints_tested: int = 0
    endpoints_passed: int = 0
    results: list[EndpointResult] = field(default_factory=list)
    error: str = ""


def extract_files(agent_result: AgentResult) -> dict[str, str]:
    """Extract file path -> content from write_file tool calls.

    Parses tool_call arguments (JSON string with 'path' and 'content' keys)
    across all turns of an agent result.
    """
    files: dict[str, str] = {}
    for turn in agent_result.turns:
        for tc in turn.tool_calls:
            func = tc.get("function", {})
            if func.get("name") != "write_file":
                continue
            args_str = func.get("arguments", "")
            if not args_str:
                continue
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
            except (json.JSONDecodeError, TypeError):
                continue
            path = args.get("path", "")
            content = args.get("content", "")
            if path and content:
                files[path] = content
    return files


def _find_server_file(files: dict[str, str], spec: ScenarioEndpoints) -> str | None:
    """Identify the server entry-point file from extracted files."""
    for path in files:
        if path.endswith(spec.server_file):
            return path
        # Also match bare filename
        if Path(path).name == spec.server_file:
            return path
    # Fallback: look for any .py file with 'http.server' or 'serve' in content
    for path, content in files.items():
        if path.endswith(".py") and ("http.server" in content or "HTTPServer" in content or "serve" in content.lower()):
            return path
    return None


def _find_free_port() -> int:
    """Find a free TCP port to avoid collisions."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, timeout: float = SERVER_START_TIMEOUT) -> bool:
    """Poll until server is accepting connections or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _rewrite_port(content: str, original_port: int, new_port: int) -> str:
    """Replace the listening port in server source code."""
    return content.replace(str(original_port), str(new_port))


def evaluate_agent(
    agent_result: AgentResult,
    scenario_id: str,
) -> SandboxResult:
    """Run sandbox evaluation for a single agent's generated code.

    1. Extract files from tool calls
    2. Write to temp directory
    3. Start the server
    4. Hit each expected endpoint
    5. Score: endpoints_passing / endpoints_total
    6. Clean up
    """
    result = SandboxResult(agent_id=agent_result.agent_id)
    spec = get_endpoint_specs(scenario_id)
    if spec is None:
        result.error = f"No endpoint specs for scenario {scenario_id}"
        return result

    files = extract_files(agent_result)
    if not files:
        result.error = "No files extracted from tool calls"
        return result

    server_file = _find_server_file(files, spec)
    if server_file is None:
        result.error = f"No server file matching '{spec.server_file}' found"
        return result

    port = _find_free_port()
    tmpdir = None
    proc = None

    try:
        tmpdir = tempfile.mkdtemp(prefix="pawbench_sandbox_")

        # Write all extracted files
        for path, content in files.items():
            full = Path(tmpdir) / path
            full.parent.mkdir(parents=True, exist_ok=True)
            # Rewrite port in server file to avoid collisions
            if path == server_file:
                content = _rewrite_port(content, spec.port, port)
            full.write_text(content)

        # Start the server
        server_path = Path(tmpdir) / server_file
        proc = subprocess.Popen(
            ["python3", str(server_path)],
            cwd=tmpdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,
        )

        if not _wait_for_server(port):
            stderr = ""
            if proc.poll() is not None:
                stderr = (proc.stderr.read() or b"").decode(errors="replace")[:500]
            result.error = f"Server did not start within {SERVER_START_TIMEOUT}s. stderr: {stderr}"
            return result

        # Test each endpoint
        base = f"http://127.0.0.1:{port}"
        for ep_spec in spec.endpoints:
            er = EndpointResult(spec=ep_spec)
            try:
                resp = requests.request(
                    ep_spec.method,
                    f"{base}{ep_spec.path}",
                    timeout=REQUEST_TIMEOUT,
                )
                er.status_code = resp.status_code
                try:
                    body = resp.json()
                except Exception:
                    body = None
                er.passed = validate_response(ep_spec, resp.status_code, body)
            except Exception as exc:
                er.error = str(exc)
            result.results.append(er)

        result.endpoints_tested = len(result.results)
        result.endpoints_passed = sum(1 for r in result.results if r.passed)
        result.score = result.endpoints_passed / max(result.endpoints_tested, 1)

    except Exception as exc:
        result.error = str(exc)
    finally:
        # Kill the server process group
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
            proc.wait(timeout=5)
        # Clean up temp dir
        if tmpdir is not None:
            import shutil

            shutil.rmtree(tmpdir, ignore_errors=True)

    return result


class SandboxEvaluator:
    """High-level evaluator that scores all agents for a scenario."""

    def __init__(self, scenario_id: str) -> None:
        self.scenario_id = scenario_id

    def evaluate(self, agents: list[AgentResult]) -> list[SandboxResult]:
        """Evaluate all agents and return sandbox results."""
        results: list[SandboxResult] = []
        for agent in agents:
            sr = evaluate_agent(agent, self.scenario_id)
            results.append(sr)
            logger.info(
                "Sandbox [%s/%s]: score=%.2f (%d/%d) %s",
                self.scenario_id,
                agent.agent_id,
                sr.score,
                sr.endpoints_passed,
                sr.endpoints_tested,
                sr.error or "OK",
            )
        return results

    def average_score(self, agents: list[AgentResult]) -> float:
        """Return the mean sandbox score across all agents."""
        results = self.evaluate(agents)
        if not results:
            return 0.0
        return sum(r.score for r in results) / len(results)

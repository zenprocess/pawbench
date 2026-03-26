# Contributing to PawBench

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/zenprocess/pawbench.git
cd pawbench
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Adding a Scenario

Scenarios are JSON files in `src/pawbench/scenarios/`. See existing PawStyle scenarios for the format.

Requirements for a good scenario:
- Multi-turn (3+ turns per agent)
- Tool calls (write_file, read_file, run_command)
- Injected tool results between turns
- Quality expectations (`expect` block with measurable criteria)
- At least one variant should include a steering/nudge event

## Pull Request Process

1. Fork the repo and create a feature branch
2. Add tests for new functionality
3. Ensure `pytest tests/ -v` passes
4. Submit a PR with a clear description

## Code Style

- Python 3.10+
- Type hints on all public functions
- No external dependencies beyond `aiohttp` and `requests`
- Keep scenarios self-contained (no external files or APIs needed)

## Reporting Issues

Please include:
- PawBench version (`pawbench --version` or `python -c "import pawbench; print(pawbench.__version__)"`)
- Endpoint type (vLLM, TGI, OpenAI, etc.)
- Model name
- Full error output

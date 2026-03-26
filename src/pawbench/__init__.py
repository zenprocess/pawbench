"""PawBench — 4-dimensional LLM inference benchmark."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pawbench")
except PackageNotFoundError:
    __version__ = "0.1.0"

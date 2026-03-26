"""ASCII art banner for PawBench CLI."""
from __future__ import annotations

from pawbench import __version__

# ANSI color codes — lavender/purple theme
_PURPLE = "\033[38;5;141m"
_LAVENDER = "\033[38;5;183m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def print_banner() -> None:
    """Print the PawBench ASCII art banner to stdout."""
    paw = f"""{_PURPLE}{_BOLD}
       /\\_/\\
      ( o.o )   {_LAVENDER}PawBench {_RESET}{_DIM}v{__version__}{_RESET}
   {_PURPLE}{_BOLD}   > ^ <    {_LAVENDER}4-dimensional LLM inference benchmark{_RESET}
   {_PURPLE}{_BOLD}  /|   |\\   {_DIM}"More bark than bite"{_RESET}
   {_PURPLE}{_BOLD} (_|   |_){_RESET}
"""
    print(paw)

"""CLI for PawBench leaderboard operations."""

from __future__ import annotations

import argparse
import sys

from pawbench.leaderboard import render_leaderboard, validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pawbench-leaderboard",
        description="PawBench leaderboard — validate and render benchmark results",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # validate subcommand
    validate_parser = subparsers.add_parser(
        "validate", help="Validate a result JSON file against the leaderboard schema"
    )
    validate_parser.add_argument("file", help="Path to the JSON result file to validate")

    # render subcommand
    render_parser = subparsers.add_parser("render", help="Render the leaderboard from a results directory")
    render_parser.add_argument("results_dir", help="Path to the directory containing result JSON files")

    args = parser.parse_args()

    if args.command == "validate":
        errors = validate_submission(args.file)
        if errors:
            print("Validation FAILED:")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)
        else:
            print("Validation passed.")

    elif args.command == "render":
        table = render_leaderboard(args.results_dir)
        print(table)


if __name__ == "__main__":
    main()

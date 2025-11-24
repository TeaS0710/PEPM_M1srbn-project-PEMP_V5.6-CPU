"""Execute a single run by delegating to ``make run``.

This script is intentionally lightweight: it translates the run specification
into a Make invocation, redirects logs, and propagates the exit code. RAM
monitoring is left as a TODO for V5.3.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


def _parse_key_vals(items: List[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid KEY=VALUE pair: {item}")
        k, v = item.split("=", 1)
        result[k] = v
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a single experiment task via make")
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument(
        "--make-var",
        action="append",
        default=[],
        help="Repeatable KEY=VALUE make variable",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Repeatable logical override key=value (flattened at call site)",
    )
    parser.add_argument("--max-ram-mb", type=int, default=None, help="Max RAM (MB) budget – reserved for future")
    parser.add_argument("--log-path", required=True)
    return parser


def _build_command(args: argparse.Namespace) -> List[str]:
    make_vars = _parse_key_vals(args.make_var)
    overrides = args.override or []
    override_str = " ".join(overrides)

    cmd = [
        "make",
        "run",
        f"STAGE={args.stage}",
        f"PROFILE={args.profile}",
    ]
    for k, v in make_vars.items():
        cmd.append(f"{k}={v}")
    if override_str:
        cmd.append(f"OVERRIDES={override_str}")
    return cmd


def run_single(args: argparse.Namespace) -> int:
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = _build_command(args)

    if args.max_ram_mb:
        # Placeholder for V5.3+ where psutil-based monitoring will be added.
        print(
            f"[run_single] max_ram_mb={args.max_ram_mb} provided but RAM monitoring is not implemented yet.",
            file=sys.stderr,
        )

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(cmd, stdout=log_file, stderr=log_file)
    return process.returncode


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    exit_code = run_single(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

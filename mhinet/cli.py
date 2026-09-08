"""Unified command dispatcher for MHINet's reproducible entry points."""

from __future__ import annotations

import argparse
import importlib
from typing import Callable


COMMAND_MODULES = {
    "preflight": "mhinet.preflight",
    "alignment": "mhinet.alignment",
    "zero-init-smoke": "mhinet.engineering",
    "gradient-audit": "mhinet.gradient_audit",
    "real-correlation-audit": "mhinet.real_correlation_audit",
    "tiny-overfit": "mhinet.tiny_overfit",
    "tiny-gate-merge": "mhinet.tiny_gate",
    "train": "mhinet.train",
    "resume": "mhinet.train",
    "evaluate": "mhinet.evaluate",
    "profile": "mhinet.profile",
    "geometry-corr-check": "mhinet.reference_checks",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mhinet",
        description="MHINet v1 implementation/training protocol v1.2",
    )
    parser.add_argument("command", choices=tuple(COMMAND_MODULES))
    args, remainder = parser.parse_known_args(argv)
    help_requested = "--help" in remainder or "-h" in remainder
    if args.command == "resume" and "--resume" not in remainder and not help_requested:
        parser.error("mhinet resume requires the train arguments plus --resume CHECKPOINT")
    module = importlib.import_module(COMMAND_MODULES[args.command])
    entry: Callable[[list[str] | None], int] = getattr(module, "main")
    return int(entry(remainder))


if __name__ == "__main__":
    raise SystemExit(main())

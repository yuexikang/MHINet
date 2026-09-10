"""Write the deterministic parent/geographic split audit used by training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mhinet.config import RuntimePaths
from mhinet.dataio.data import grouping_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--geo-cell-degrees", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    runtime = RuntimePaths.from_json(args.runtime)
    report = grouping_audit(runtime.data_root, args.geo_cell_degrees)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

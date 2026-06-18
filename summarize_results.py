from __future__ import annotations

import argparse
import json
from pathlib import Path

from sa_clip_nm.pipeline import write_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    results = []
    for path in sorted(output_dir.glob("*/metrics.json")):
        with path.open("r", encoding="utf-8") as handle:
            results.append(json.load(handle))
    write_summary(results, output_dir)
    print(f"Summarized {len(results)} categories in {output_dir}")


if __name__ == "__main__":
    main()


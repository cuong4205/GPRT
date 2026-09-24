from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


REPLACED = {"gp", "transformer_rl"}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def merge(main_dir: Path, patch_dir: Path) -> None:
    main = json.loads((main_dir / "results.json").read_text(encoding="utf-8"))
    patch = json.loads((patch_dir / "results.json").read_text(encoding="utf-8"))
    main["protocol"]["config"] = patch["protocol"]["config"]
    for key, method_field in (
        ("searches", "method"),
        ("summary_by_seed", "method"),
        ("test_results", "method"),
    ):
        retained = [row for row in main[key] if row[method_field] not in REPLACED]
        replacement = [row for row in patch[key] if row[method_field] in REPLACED]
        main[key] = sorted(
            retained + replacement,
            key=lambda row: (row["method"], row["seed"], row.get("instance", "")),
        )
    (main_dir / "results.json").write_text(
        json.dumps(main, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(main_dir / "test_results.csv", main["test_results"])
    write_csv(main_dir / "summary_by_seed.csv", main["summary_by_seed"])
    write_csv(main_dir / "search_metrics.csv", main["searches"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("main_dir", type=Path)
    parser.add_argument("patch_dir", type=Path)
    args = parser.parse_args()
    merge(args.main_dir, args.patch_dir)

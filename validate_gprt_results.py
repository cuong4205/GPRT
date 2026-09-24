from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_METHODS = {
    "edd",
    "gp",
    "transformer_rl",
    "gprt",
    "gprt_no_transformer_seeding",
    "gprt_no_elite_imitation",
}
EXPECTED_SEEDS = set(range(20260914, 20260919))
NEURAL_METHODS = EXPECTED_METHODS - {"edd", "gp"}


def validate(root: Path) -> dict[str, object]:
    payload = json.loads((root / "results.json").read_text(encoding="utf-8"))
    split = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
    cycles = int(payload["protocol"]["config"]["cycles"])
    legacy = cycles == 4
    test_skipped = payload["protocol"]["test_policy"].startswith("skipped")
    methods = set(payload["protocol"]["methods"])
    seeds = set(payload["protocol"]["seeds"])
    assert payload["protocol"]["split_counts"] == {"train": 32, "validation": 12, "test": 12}
    screening_run = bool(payload["protocol"].get("screening_run", False))
    if screening_run:
        assert test_skipped
        assert 1 <= len(seeds) <= 2
    else:
        assert seeds == EXPECTED_SEEDS
    if legacy:
        assert methods == EXPECTED_METHODS
    else:
        assert cycles >= math.ceil(32 / int(payload["protocol"]["config"]["batch_size_instances"]))
        assert payload["protocol"]["config"]["require_full_train_coverage"]
        assert payload["protocol"]["config"]["screening_train_size"] > 0
        assert payload["protocol"]["config"]["advantage_mode"] in {
            "batch_standardize", "ewma_scale", "none"
        }
    assert "Gehring" not in json.dumps(payload["protocol"])

    rows = payload["test_results"]
    assert len(rows) == (0 if test_skipped else len(methods) * len(seeds) * 12)
    keys = [(row["method"], row["seed"], row["instance"]) for row in rows]
    assert len(keys) == len(set(keys))
    test_names = {name.upper() for name, part in split.items() if part == "test"}
    assert not rows or {row["instance"] for row in rows} == test_names
    assert all(row["feasible"] or row["unserved_count"] > 0 for row in rows)
    assert all(
        math.isfinite(float(row[field]))
        for row in rows
        for field in ("vehicles", "distance", "normalized_distance", "runtime_ms")
    )

    searches = {(row["method"], row["seed"]): row for row in payload["searches"]}
    assert set(method for method, _ in searches) == methods
    assert all({seed for method, seed in searches if method == name} == seeds for name in methods)
    if legacy:
        assert all(
            int(searches[(method, seed)]["fitness_requests"]) == 1424
            for method in methods - {"edd"}
            for seed in seeds
        )
    else:
        assert all(
            int(searches[(method, seed)]["fitness_requests"]) > 0
            for method in methods - {"edd"}
            for seed in seeds
        )
    neural_methods = methods - {"edd", "gp"}
    for method in methods - {"edd"}:
        for seed in seeds:
            run_dir = root / method / str(seed)
            expression = (run_dir / "final_heuristic.txt").read_text(encoding="utf-8").strip()
            assert expression == searches[(method, seed)]["expression"]
            if not test_skipped:
                assert {row["expression"] for row in rows if row["method"] == method and row["seed"] == seed} == {expression}
            if method in neural_methods:
                assert (run_dir / "best.pt").is_file()
                assert (run_dir / "last.pt").is_file()
                config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
                assert config["use_gp"] == (method != "transformer_rl")
                assert config["use_transformer_seeding"] == (method != "gprt_no_transformer_seeding")
                assert config["use_elite_imitation"] == (method not in {"transformer_rl", "gprt_no_elite_imitation"})
                assert config.get("archive_seed_count", 0) == (
                    4 if method == "gprt_archive_seeding" else 0
                )
                run_split = json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8"))
                assert run_split == split
                if not legacy:
                    summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
                    assert summary["training_coverage_count"] == 32
                    assert summary["training_coverage_rate"] == 1.0
                    assert len(summary["screening_instance_names"]) == config["screening_train_size"]
                    archive = json.loads((run_dir / "candidate_archive.json").read_text(encoding="utf-8"))
                    assert archive and all("observations" in item for item in archive)

    return {
        "status": "valid",
        "protocol": "legacy_4cycle" if legacy else "improved",
        "methods": len(methods),
        "training_seeds_per_method": len(seeds),
        "test_rows": len(rows),
        "test_skipped": test_skipped,
        "screening_run": screening_run,
        "split_counts": payload["protocol"]["split_counts"],
        "neural_checkpoints": len(neural_methods) * len(seeds) * 2,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/gprt_vrptw/reduced_main"))
    args = parser.parse_args()
    print(json.dumps(validate(args.results_dir), indent=2))

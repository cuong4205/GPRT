"""Summarize all preregistered screening runs, including unfavorable outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def analyze(root: Path) -> dict:
    variants = ("baseline", "routing_rank", "elite3")
    seeds = (20260914, 20260915)
    runs = {}
    reference_hashes = None
    reference_data = None
    reference_split = None
    reference_config = None
    for variant in variants:
        for seed in seeds:
            folder = root / variant / str(seed)
            summary = json.loads((folder / "quality_summary.json").read_text(encoding="utf-8"))
            metadata = json.loads((folder / "audit_metadata.json").read_text(encoding="utf-8"))
            if reference_hashes is None:
                reference_hashes = metadata["code_sha256"]
                reference_data = metadata["instance_sha256"]
                reference_split = metadata["split"]
                reference_config = metadata["config"]
            assert reference_hashes == metadata["code_sha256"]
            assert reference_data == metadata["instance_sha256"]
            assert reference_split == metadata["split"]
            changed_config = {key: value for key, value in metadata["config"].items()
                              if value != reference_config[key]}
            assert changed_config == ({"elite_fraction": 3 / 64} if variant == "elite3" else {})
            assert summary["test_evaluated"] is False
            assert summary["training_coverage"] == 1.0 and summary["gradient_steps"] == 8
            assert len(summary["validation_rows"]) == 12
            assert summary["training_stats"]["train_instance_requests"] == 11264
            assert len(summary["reward_diagnostics"]) == 8
            assert all(row["samples"] == 32 for row in summary["reward_diagnostics"])
            runs[variant, seed] = summary

    comparisons = []
    for variant in variants[1:]:
        per_seed = []
        for seed in seeds:
            baseline, candidate = runs["baseline", seed], runs[variant, seed]
            base_fitness = tuple(baseline["validation_fitness"][:4])
            new_fitness = tuple(candidate["validation_fitness"][:4])
            outcome = "win" if new_fitness < base_fitness else "loss" if new_fitness > base_fitness else "tie"
            base_rows = {r["instance"]: r for r in baseline["validation_rows"]}
            instance_outcomes = {"win": 0, "loss": 0, "tie": 0}
            for row in candidate["validation_rows"]:
                old = base_rows[row["instance"]]
                a = (not row["feasible"], row["unserved"], row["vehicles"], row["distance"])
                b = (not old["feasible"], old["unserved"], old["vehicles"], old["distance"])
                instance_outcomes["win" if a < b else "loss" if a > b else "tie"] += 1
            per_seed.append({
                "seed": seed, "outcome": outcome, "baseline_fitness": base_fitness,
                "candidate_fitness": new_fitness,
                "delta_vehicles": new_fitness[2] - base_fitness[2],
                "delta_normalized_distance": new_fitness[3] - base_fitness[3],
                "instance_outcomes_descriptive_only": instance_outcomes,
            })
        comparisons.append({
            "variant": variant,
            "eligible_for_confirmation": all(r["outcome"] == "win" and r["candidate_fitness"][0] <= r["baseline_fitness"][0] for r in per_seed),
            "per_seed": per_seed,
        })

    rows = []
    for (variant, seed), run in runs.items():
        diagnostic = run["reward_diagnostics"]
        rows.append({
            "variant": variant, "seed": seed,
            "validation_fitness": run["validation_fitness"], "expression": run["expression"],
            "complexity_only_pair_fraction": sum(r["pairs_separated_only_by_complexity"] for r in diagnostic) / sum(r["all_pairs"] for r in diagnostic),
            "end_to_end_train_seconds": run["end_to_end_train_seconds"],
            "peak_rss_mb": run["peak_rss_mb"],
            "logical_instance_evaluations": run["training_stats"]["instance_evaluations"],
            "physical_instance_evaluations": run.get("physical_instance_evaluations", run["training_stats"]["instance_evaluations"]),
            "replayed_logical_evaluations": run.get("replayed_logical_evaluations", 0),
        })
    result = {
        "status": "screening only; two training seeds, validation selection, no test evaluation",
        "acceptance": "Both seeds must improve lexicographic routing fitness and not reduce feasibility",
        "comparisons": comparisons, "runs": rows,
        "limitations": [
            "Two seeds do not establish statistical reliability or out-of-distribution generalization.",
            "Per-instance outcomes are descriptive and not independent training runs.",
            "Replay and concurrent execution make runtime unsuitable for cold-speed comparisons.",
        ],
    }
    (root / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/gprt_vrptw/quality_screening_20260917"))
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), indent=2))

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from vrptw_baselines.data import VRPTWInstance, discover_instances
from vrptw_baselines.gp import GPConfig, evolve, make_priority, tree_size
from vrptw_baselines.solver import Solution, solve


METHODS = ("FIFO", "Random", "Greedy", "EDD", "GP")


def family(name: str) -> str:
    stem = name.lower()
    prefix = stem.split("_")[0]
    letters = "".join(character for character in prefix if character.isalpha())
    digits = "".join(character for character in prefix if character.isdigit())
    return f"{letters.upper()}{digits[:1]}"


def split_solomon(instances: list[VRPTWInstance]) -> dict[str, str]:
    """Stratified deterministic 60/20/20 split within C1/C2/R1/R2/RC1/RC2."""
    grouped: dict[str, list[VRPTWInstance]] = defaultdict(list)
    for instance in instances:
        grouped[family(instance.name)].append(instance)
    split: dict[str, str] = {}
    for group_instances in grouped.values():
        ordered = sorted(
            group_instances,
            key=lambda item: hashlib.sha256(item.name.lower().encode("utf-8")).hexdigest(),
        )
        validation_count = max(1, round(len(ordered) * 0.2))
        test_count = max(1, round(len(ordered) * 0.2))
        train_end = len(ordered) - validation_count - test_count
        validation_end = len(ordered) - test_count
        for index, instance in enumerate(ordered):
            split[instance.name.lower()] = (
                "train" if index < train_end else "validation" if index < validation_end else "test"
            )
    return split


def select_gp_train(instances: list[VRPTWInstance], limit: int) -> list[VRPTWInstance]:
    grouped: dict[str, list[VRPTWInstance]] = defaultdict(list)
    for instance in sorted(instances, key=lambda item: item.name.lower()):
        grouped[family(instance.name)].append(instance)
    selected: list[VRPTWInstance] = []
    depth = 0
    while len(selected) < limit:
        added = False
        for group_name in sorted(grouped):
            if depth < len(grouped[group_name]):
                selected.append(grouped[group_name][depth])
                added = True
                if len(selected) == limit:
                    return selected
        if not added:
            break
        depth += 1
    return selected


def solution_row(
    instance: VRPTWInstance,
    solution: Solution,
    *,
    run: int,
    seed: int,
    split: str,
    runtime_ms: float,
    expression: str = "",
) -> dict[str, object]:
    return {
        "run": run,
        "seed": seed,
        "method": solution.method.upper() if solution.method != "greedy" else "Greedy",
        "instance": instance.name.upper(),
        "family": family(instance.name),
        "split": split,
        "source": instance.source,
        "customers": instance.customer_count,
        "max_vehicles": instance.max_vehicles,
        "capacity": instance.capacity,
        "feasible": solution.feasible,
        "vehicle_count": solution.vehicle_count,
        "distance": solution.distance,
        "waiting_time": solution.waiting_time,
        "objective": solution.objective,
        "runtime_ms": runtime_ms,
        "unserved_count": len(solution.unserved),
        "unserved": ",".join(map(str, solution.unserved)),
        "expression": expression,
    }


def route_rows(
    instance: VRPTWInstance,
    solution: Solution,
    *,
    run: int,
    seed: int,
) -> list[dict[str, object]]:
    return [
        {
            "run": run,
            "seed": seed,
            "method": solution.method.upper() if solution.method != "greedy" else "Greedy",
            "instance": instance.name.upper(),
            "route_id": route.route_id,
            "route": "-".join(map(str, route.customers)),
            "customer_visits": max(0, len(route.customers) - 2),
            "load": route.load,
            "distance": route.distance,
            "end_time": route.end_time,
            "waiting_time": route.waiting_time,
            "feasible": route.feasible,
        }
        for route in solution.route_metrics
    ]


def summarize(details: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    test_rows = [row for row in details if row["split"] == "test"]
    by_run: dict[tuple[str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in test_rows:
        by_run[(str(row["method"]), int(row["run"]), int(row["seed"]))].append(row)

    run_summary: list[dict[str, object]] = []
    for (method, run, seed), rows in sorted(by_run.items()):
        feasible = [row for row in rows if row["feasible"]]
        metric_rows = feasible if feasible else rows
        run_summary.append(
            {
                "method": method,
                "run": run,
                "seed": seed,
                "test_instances": len(rows),
                "feasible_rate": len(feasible) / len(rows),
                "mean_vehicles": statistics.fmean(float(row["vehicle_count"]) for row in metric_rows),
                "mean_distance": statistics.fmean(float(row["distance"]) for row in metric_rows),
                "mean_waiting_time": statistics.fmean(float(row["waiting_time"]) for row in metric_rows),
                "mean_objective": statistics.fmean(float(row["objective"]) for row in rows),
                "total_runtime_ms": sum(float(row["runtime_ms"]) for row in rows),
            }
        )

    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in run_summary:
        by_method[str(row["method"])].append(row)
    method_summary: list[dict[str, object]] = []
    for method, rows in by_method.items():
        record: dict[str, object] = {"method": method, "runs": len(rows)}
        for metric in (
            "feasible_rate",
            "mean_vehicles",
            "mean_distance",
            "mean_waiting_time",
            "mean_objective",
            "total_runtime_ms",
        ):
            values = [float(row[metric]) for row in rows]
            record[f"{metric}_mean"] = statistics.fmean(values)
            record[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        method_summary.append(record)
    method_summary.sort(
        key=lambda row: (
            -float(row["feasible_rate_mean"]),
            float(row["mean_vehicles_mean"]),
            float(row["mean_distance_mean"]),
        )
    )
    for rank, row in enumerate(method_summary, start=1):
        row["rank"] = rank
    return run_summary, method_summary


def inventory_row(instance: VRPTWInstance, split: str) -> dict[str, object]:
    return {
        "instance": instance.name.upper(),
        "source": instance.source,
        "family": family(instance.name),
        "customers": instance.customer_count,
        "max_vehicles": instance.max_vehicles,
        "capacity": instance.capacity,
        "depot_due_date": instance.depot.due_date,
        "split": split,
        "sha256": instance.sha256,
        "path": str(instance.path),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    solomon = discover_instances(args.data_dir)
    unexpected = [instance for instance in solomon if instance.source != "Solomon"]
    if unexpected:
        raise RuntimeError("The selected data directory contains non-Solomon instances")
    if not solomon:
        raise RuntimeError("No Solomon instances found")
    split_map = split_solomon(solomon)
    train = [instance for instance in solomon if split_map[instance.name.lower()] == "train"]
    validation = [instance for instance in solomon if split_map[instance.name.lower()] == "validation"]
    evaluation = sorted(solomon, key=lambda item: item.name.lower())
    gp_train = select_gp_train(train, args.gp_train_limit)
    config = GPConfig(
        population_size=args.gp_population,
        generations=args.gp_generations,
        max_depth=args.gp_max_depth,
    )

    details: list[dict[str, object]] = []
    routes: list[dict[str, object]] = []
    gp_runs: list[dict[str, object]] = []
    for run_index in range(1, args.runs + 1):
        seed = args.base_seed + run_index - 1
        gp_started = time.perf_counter()
        gp_result = evolve(gp_train, validation, seed=seed, config=config)
        gp_search_seconds = time.perf_counter() - gp_started
        gp_runs.append(
            {
                "run": run_index,
                "seed": seed,
                "expression": gp_result.expression_text,
                "tree_size": tree_size(gp_result.expression),
                "train_fitness": gp_result.train_fitness,
                "validation_fitness": gp_result.validation_fitness,
                "fitness_requests": gp_result.fitness_requests,
                "unique_expressions": gp_result.unique_expressions,
                "instance_evaluations": gp_result.instance_evaluations,
                "cache_hits": gp_result.cache_hits,
                "gp_search_seconds": gp_search_seconds,
                "generation_best": list(gp_result.generation_best),
            }
        )
        gp_priority = make_priority(gp_result.expression)

        for instance in evaluation:
            split = split_map[instance.name.lower()]
            for method in ("fifo", "random", "greedy", "edd", "gp"):
                started = time.perf_counter()
                solution = solve(
                    instance,
                    method,
                    seed=seed,
                    priority_function=gp_priority if method == "gp" else None,
                )
                runtime_ms = (time.perf_counter() - started) * 1000.0
                details.append(
                    solution_row(
                        instance,
                        solution,
                        run=run_index,
                        seed=seed,
                        split=split,
                        runtime_ms=runtime_ms,
                        expression=gp_result.expression_text if method == "gp" else "",
                    )
                )
                routes.extend(route_rows(instance, solution, run=run_index, seed=seed))

    run_summary, method_summary = summarize(details)
    inventory = [
        inventory_row(
            instance,
            split_map.get(instance.name.lower(), "inventory_only"),
        )
        for instance in solomon
    ]
    return {
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "data_dir": str(Path(args.data_dir).resolve()),
            "all_instance_count": len(solomon),
            "solomon_instance_count": len(solomon),
            "train_count": len(train),
            "validation_count": len(validation),
            "test_count": sum(value == "test" for value in split_map.values()),
            "evaluation_scope": "All Solomon instances; primary summary uses only fixed test split",
            "objective": "Lexicographic: feasible, then vehicles, then Euclidean distance",
            "distance_rule": "Unrounded Euclidean distance",
            "runs": args.runs,
            "base_seed": args.base_seed,
            "gp_train_limit": args.gp_train_limit,
            "gp_train_instances": [instance.name.upper() for instance in gp_train],
            "gp_config": asdict(config),
        },
        "inventory": inventory,
        "gp_runs": gp_runs,
        "details": details,
        "routes": routes,
        "run_summary": run_summary,
        "method_summary": method_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible VRPTW heuristic and GP baselines")
    parser.add_argument("--data-dir", type=Path, default=Path("data/data/Solomon"))
    parser.add_argument("--output", type=Path, default=Path("outputs/vrptw_baselines/results.json"))
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=20260914)
    parser.add_argument("--gp-train-limit", type=int, default=6)
    parser.add_argument("--gp-population", type=int, default=10)
    parser.add_argument("--gp-generations", type=int, default=4)
    parser.add_argument("--gp-max-depth", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.runs < 5:
        raise SystemExit("--runs must be at least 5 for the baseline protocol")
    payload = run(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["method_summary"], ensure_ascii=False, indent=2))

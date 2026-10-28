"""Paired Python/Numba benchmark, fixed train-only workload, no fitness cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from gprt_vrptw.evaluator import make_priority
from gprt_vrptw.grammar import FEATURES, Grammar, tree_to_string
from gprt_vrptw.splits import split_solomon, stratified_sample
from vrptw_baselines.data import discover_instances
from vrptw_baselines.solver import solve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--before-solutions", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics/route_builder_benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    instances = discover_instances("data/data/Solomon")
    split = split_solomon(instances)
    train = stratified_sample([i for i in instances if split[i.name.lower()] == "train"], 6)
    rng = random.Random(2810)
    grammar = Grammar(63, 6)
    trees = [*FEATURES, *(grammar.random_tree(rng) for _ in range(16)),
             ("sub", "d_current", "next_feasible_fraction"),
             ("add", "next_feasible_fraction", "urgency_rank")]
    before = {}
    if args.before_solutions:
        before = {(row["instance"], row["expression"]): row["solution"]
                  for row in json.loads(args.before_solutions.read_text())}
    started = time.perf_counter()
    solve(train[0], "gp", priority_function=make_priority(trees[0], backend="numba"))
    first_call = time.perf_counter() - started
    rows = []
    for tree in trees:
        priorities = {mode: make_priority(tree, backend=mode) for mode in ("python", "numba")}
        for instance in train:
            reference = solve(instance, "gp", priority_function=priorities["python"])
            native = solve(instance, "gp", priority_function=priorities["numba"])
            if reference != native:
                raise AssertionError(f"Solution changed: {instance.name}, {tree}")
            key = instance.name, tree_to_string(tree)
            serialized = json.loads(json.dumps(asdict(reference)))
            if before and serialized != before[key]:
                raise AssertionError(f"Solution differs from before implementation: {key}")
            times = {mode: [] for mode in priorities}
            for _ in range(args.repeats):
                order = list(priorities)
                rng.shuffle(order)
                for mode in order:
                    started = time.perf_counter()
                    result = solve(instance, "gp", priority_function=priorities[mode])
                    times[mode].append(time.perf_counter() - started)
                    if result != reference:
                        raise AssertionError(f"Nondeterministic solve: {key}, {mode}")
            medians = {mode: statistics.median(values) for mode, values in times.items()}
            rows.append({"instance": instance.name, "expression": key[1], "seconds": times,
                         "median_seconds": medians, "speedup": medians["python"] / medians["numba"],
                         "solution_sha256": hashlib.sha256(json.dumps(serialized, sort_keys=True).encode()).hexdigest()})
    totals = {mode: sum(row["median_seconds"][mode] for row in rows) for mode in ("python", "numba")}
    report = {"split": "train only, one instance per family", "instances": [i.name for i in train],
              "seed": 2810, "expressions": len(trees), "pairs": len(rows), "repeats": args.repeats,
              "python_version": platform.python_version(), "platform": platform.platform(),
              "numpy": version("numpy"), "numba": version("numba"), "llvmlite": version("llvmlite"),
              "first_numba_call_seconds_including_import_and_disk_cache_or_compile": first_call,
              "all_solutions_identical": True, "checked_against_before_snapshot": bool(before),
              "sum_median_seconds": totals, "aggregate_speedup": totals["python"] / totals["numba"],
              "per_case_speedup_median": statistics.median(row["speedup"] for row in rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()

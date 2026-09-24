"""Compare dense and lazy feature evaluation on train only, without fitness cache.

Run from the project root: python -m scripts.benchmark_lazy_features
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

from gprt_vrptw.evaluator import compile_tree, make_priority, normalized_feature_vector
from gprt_vrptw.grammar import tree_to_string
from gprt_vrptw.splits import split_solomon, stratified_sample
from vrptw_baselines.data import discover_instances
from vrptw_baselines.solver import solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics/lazy_features_benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    instances = discover_instances(Path("data/data/Solomon"))
    split = split_solomon(instances)
    train = stratified_sample([i for i in instances if split[i.name.lower()] == "train"], 6)
    trees = ["d_current", ("add", "arrival_time", "wait_time"),
             ("if_else", ("le", "slack", "wait_time"), "due_date", "d_current"),
             ("sub", "d_current", "next_feasible_fraction")]
    rng = random.Random(42)
    rows = []
    for tree in trees:
        compiled = compile_tree(tree)
        def dense(instance, context):
            return compiled(normalized_feature_vector(instance, context))
        functions = {"dense": dense, "lazy": make_priority(tree)}
        for instance in train:
            for function in functions.values():
                solve(instance, "gp", priority_function=function)
            durations = {mode: [] for mode in functions}
            reference = None
            for _ in range(args.repeats):
                order = list(functions)
                rng.shuffle(order)
                for mode in order:
                    start = time.perf_counter()
                    result = solve(instance, "gp", priority_function=functions[mode])
                    durations[mode].append(time.perf_counter() - start)
                    if reference is None:
                        reference = result
                    if result != reference:
                        raise AssertionError(f"Solution changed for {instance.name}: {tree}")
            medians = {mode: statistics.median(times) for mode, times in durations.items()}
            row = {"instance": instance.name, "expression": tree_to_string(tree),
                   "seconds": durations, "median_seconds": medians,
                   "speedup": medians["dense"] / medians["lazy"], "identical_solution": True}
            rows.append(row)
            print(json.dumps({key: value for key, value in row.items() if key != "seconds"}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"seed": 42, "repeats": args.repeats,
        "split": "train only; one instance per family", "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

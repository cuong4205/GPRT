from __future__ import annotations

import argparse
import ctypes
import csv
import json
import os
import random
import statistics
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from vrptw_baselines.data import VRPTWInstance, discover_instances
from vrptw_baselines.solver import solve

from gprt_vrptw.config import GPRTConfig
from gprt_vrptw.evaluator import Fitness, FitnessEvaluator, distance_scale
from gprt_vrptw.gp import evolve_population
from gprt_vrptw.grammar import Grammar, Tree, tree_to_string
from gprt_vrptw.selection import record_observation, screen_archive
from gprt_vrptw.splits import family, split_solomon, stratified_batches, stratified_sample
from gprt_vrptw.trainer import GPRTTrainer


METHODS = (
    "edd",
    "gp",
    "transformer_rl",
    "gprt",
    "gprt_no_transformer_seeding",
    "gprt_no_elite_imitation",
    "gprt_alpha_001",
    "gprt_alpha_005",
    "gprt_typed_grammar",
    "gprt_no_lookahead_features",
    "gprt_archive_seeding",
)
CORE_METHODS = METHODS[:6]


def current_process_rss_bytes() -> int:
    """Current resident memory without an optional third-party dependency."""
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return 0
        return int(counters.WorkingSetSize)
    try:
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (FileNotFoundError, IndexError, OSError, ValueError):
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return peak if os.uname().sysname == "Darwin" else peak * 1024


class MemorySampler:
    def __init__(self, interval_seconds: float = 0.1):
        self.interval_seconds = interval_seconds
        self.peak_bytes = current_process_rss_bytes()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self.peak_bytes = max(self.peak_bytes, current_process_rss_bytes())

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> int:
        self.stop_event.set()
        self.thread.join()
        self.peak_bytes = max(self.peak_bytes, current_process_rss_bytes())
        return self.peak_bytes


def method_config(base: GPRTConfig, method: str) -> GPRTConfig:
    if method == "gprt":
        return replace(base)
    if method == "transformer_rl":
        return replace(base, use_gp=False, use_elite_imitation=False)
    if method == "gprt_no_transformer_seeding":
        return replace(base, use_transformer_seeding=False)
    if method == "gprt_no_elite_imitation":
        return replace(base, use_elite_imitation=False)
    if method == "gprt_alpha_001":
        return replace(base, alpha=0.01)
    if method == "gprt_alpha_005":
        return replace(base, alpha=0.05)
    if method == "gprt_typed_grammar":
        return replace(base, typed_grammar=True)
    if method == "gprt_no_lookahead_features":
        return replace(base, use_lookahead_features=False)
    if method == "gprt_archive_seeding":
        return replace(base, archive_seed_count=4)
    raise ValueError(method)


def run_gp_baseline(
    train: list[VRPTWInstance],
    validation: list[VRPTWInstance],
    config: GPRTConfig,
    seed: int,
) -> tuple[Tree, Fitness, dict[str, Any]]:
    rng = random.Random(seed)
    grammar = Grammar(
        config.max_tokens,
        config.max_depth,
        typed=config.typed_grammar,
        include_lookahead=config.use_lookahead_features,
    )
    evaluator = FitnessEvaluator(
        config.complexity_penalty,
        use_lookahead_features=config.use_lookahead_features,
    )
    batches = stratified_batches(train, config.batch_size_instances, seed=seed)
    if config.require_full_train_coverage and config.cycles < len(batches):
        raise ValueError(
            f"cycles={config.cycles} covers only {config.cycles}/{len(batches)} training batches"
        )
    screening = stratified_sample(train, config.screening_train_size)
    archive: dict[str, dict[str, Any]] = {}
    best_tree: Tree | None = None
    best_validation: Fitness | None = None
    started = time.perf_counter()
    for cycle in range(1, config.cycles + 1):
        seeds = [grammar.random_tree(rng, grow=index % 2 == 0) for index in range(config.population_size)]
        population, _, _ = evolve_population(
            seeds,
            batches[(cycle - 1) % len(batches)],
            evaluator,
            rng,
            grammar,
            replace(config, use_gp=True),
        )
        batch = batches[(cycle - 1) % len(batches)]
        for tree in population:
            fitness = evaluator.fitness(tree, batch)
            record_observation(archive, tree, fitness, cycle, batch)
        # Match GPRT's additional N=32 on-policy fitness requests. GP receives
        # these as ordinary random exploration candidates, so the budget is both
        # equal and usable by the baseline search.
        for _ in range(config.transformer_samples):
            tree = grammar.random_tree(rng, grow=True)
            fitness = evaluator.fitness(tree, batch)
            record_observation(archive, tree, fitness, cycle, batch)
        if cycle % config.validation_interval == 0 or cycle == config.cycles:
            candidates = screen_archive(
                archive, evaluator, screening, config.validation_candidates
            )
            for _, item in candidates:
                tree = item["tree"]
                fitness = evaluator.fitness(tree, validation, phase="validation")
                if best_validation is None or fitness < best_validation:
                    best_tree, best_validation = tree, fitness
    assert best_tree is not None and best_validation is not None
    return best_tree, best_validation, {
        "fitness_requests": evaluator.stats.fitness_requests,
        "unique_evaluations": evaluator.stats.unique_evaluations,
        "instance_evaluations": evaluator.stats.instance_evaluations,
        "train_instance_requests": evaluator.stats.train_instance_requests,
        "screening_instance_requests": evaluator.stats.screening_instance_requests,
        "validation_instance_requests": evaluator.stats.validation_instance_requests,
        "diagnostic_instance_requests": evaluator.stats.diagnostic_instance_requests,
        "train_instance_evaluations": evaluator.stats.train_instance_evaluations,
        "screening_instance_evaluations": evaluator.stats.screening_instance_evaluations,
        "validation_instance_evaluations": evaluator.stats.validation_instance_evaluations,
        "diagnostic_instance_evaluations": evaluator.stats.diagnostic_instance_evaluations,
        "cache_hits": evaluator.stats.cache_hits,
        "search_seconds": time.perf_counter() - started,
        "archive_size": len(archive),
    }


def test_rows(method: str, seed: int, tree: Tree | None, instances: list[VRPTWInstance]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    evaluator = FitnessEvaluator()
    for instance in sorted(instances, key=lambda item: item.name.lower()):
        started = time.perf_counter()
        if method == "edd":
            solution = solve(instance, "edd", seed=seed)
            expression = "EDD(due_date, customer_id)"
        else:
            if tree is None:
                raise ValueError(f"{method} requires a locked validation heuristic")
            solution = evaluator.solution(tree, instance)
            expression = tree_to_string(tree)
        rows.append(
            {
                "method": method,
                "seed": seed,
                "instance": instance.name.upper(),
                "family": family(instance.name),
                "feasible": solution.feasible,
                "unserved_count": len(solution.unserved),
                "vehicles": solution.vehicle_count,
                "distance": solution.distance,
                "normalized_distance": solution.distance / (instance.customer_count * distance_scale(instance)),
                "runtime_ms": (time.perf_counter() - started) * 1000,
                "expression": expression,
            }
        )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["method"]), int(row["seed"])), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (method, seed), values in sorted(groups.items()):
        summaries.append(
            {
                "method": method,
                "seed": seed,
                "test_instances": len(values),
                "feasibility_rate": statistics.fmean(float(v["feasible"]) for v in values),
                "mean_unserved": statistics.fmean(float(v["unserved_count"]) for v in values),
                "mean_vehicles": statistics.fmean(float(v["vehicles"]) for v in values),
                "mean_distance": statistics.fmean(float(v["distance"]) for v in values),
                "mean_normalized_distance": statistics.fmean(float(v["normalized_distance"]) for v in values),
                "total_runtime_ms": sum(float(v["runtime_ms"]) for v in values),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate GPRT-VRPTW and fair ablations")
    parser.add_argument("--data-dir", type=Path, default=Path("data/data/Solomon"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gprt_vrptw/main"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(CORE_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20260914, 20260919)))
    parser.add_argument("--cycles", type=int, default=32)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--transformer-samples", type=int, default=32)
    parser.add_argument("--gp-generations", type=int, default=3)
    parser.add_argument("--screening-train-size", type=int, default=8)
    parser.add_argument(
        "--screening-run",
        action="store_true",
        help="Allow 1-2 seeds for train/validation screening before the main protocol",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Train/select on train+validation only; do not evaluate the held-out test split",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.screening_run and len(args.seeds) < 5:
        raise SystemExit("The main protocol requires at least five independent training seeds")
    if args.screening_run and len(args.seeds) > 2:
        raise SystemExit("A screening run uses at most two seeds")

    instances = discover_instances(args.data_dir)
    if len(instances) != 56 or any(instance.source != "Solomon" for instance in instances):
        raise SystemExit("Expected exactly 56 Solomon instances and no Gehring-Homberger data")
    split = split_solomon(instances)
    train = [i for i in instances if split[i.name.lower()] == "train"]
    validation = [i for i in instances if split[i.name.lower()] == "validation"]
    test = [i for i in instances if split[i.name.lower()] == "test"]
    config = GPRTConfig(
        cycles=args.cycles,
        population_size=args.population,
        transformer_samples=args.transformer_samples,
        gp_generations=args.gp_generations,
        screening_train_size=args.screening_train_size,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict[str, Any]] = []
    searches: list[dict[str, Any]] = []

    for method in args.methods:
        for seed in args.seeds:
            run_dir = args.output_dir / method / str(seed)
            run_dir.mkdir(parents=True, exist_ok=True)
            run_started = time.perf_counter()
            memory_sampler = MemorySampler()
            memory_sampler.start()
            if method == "edd":
                tree = None
                validation_fitness = None
                search = {"search_seconds": 0.0}
            elif method == "gp":
                tree, validation_fitness, search = run_gp_baseline(train, validation, config, seed)
                (run_dir / "final_heuristic.txt").write_text(tree_to_string(tree) + "\n", encoding="utf-8")
            else:
                run_config = method_config(config, method)
                trainer = GPRTTrainer(run_config, device=args.device)
                search_started = time.perf_counter()
                result = trainer.train(
                    train,
                    validation,
                    seed=seed,
                    output_dir=run_dir,
                    split_manifest=split,
                )
                tree = result.best_tree
                validation_fitness = result.best_validation_fitness
                search = {
                    **asdict(trainer.evaluator.stats),
                    # Includes neural work, GP, evaluator, checkpoint and final artifacts.
                    "search_seconds": time.perf_counter() - search_started,
                }
            peak_rss_bytes = memory_sampler.stop()
            search["end_to_end_seconds"] = time.perf_counter() - run_started
            search["peak_rss_mb"] = peak_rss_bytes / (1024 * 1024)
            # This is the only point where test instances may enter the run,
            # after validation has locked the checkpoint/heuristic above.
            if not args.skip_test:
                raw_rows.extend(test_rows(method, seed, tree, test))
            searches.append(
                {
                    "method": method,
                    "seed": seed,
                    "validation_fitness": list(validation_fitness) if validation_fitness else None,
                    "expression": tree_to_string(tree) if tree is not None else "EDD(due_date, customer_id)",
                    **search,
                }
            )

    summaries = aggregate(raw_rows)
    payload = {
        "protocol": {
            "data": "56 Solomon VRPTW instances only",
            "distance": "unrounded Euclidean",
            "split_counts": {part: sum(v == part for v in split.values()) for part in ("train", "validation", "test")},
            "test_policy": (
                "skipped; train and validation only"
                if args.skip_test
                else "used once after validation locks each checkpoint/heuristic"
            ),
            "seeds": args.seeds,
            "screening_run": args.screening_run,
            "config": config.to_dict(),
            "methods": args.methods,
        },
        "searches": searches,
        "summary_by_seed": summaries,
        "test_results": raw_rows,
    }
    (args.output_dir / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "split_manifest.json").write_text(json.dumps(split, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(args.output_dir / "test_results.csv", raw_rows)
    write_csv(args.output_dir / "summary_by_seed.csv", summaries)
    write_csv(args.output_dir / "search_metrics.csv", searches)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

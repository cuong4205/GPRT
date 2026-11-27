"""Train GP/GPRT on dynamic CSV episodes and replay locked expressions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from dvrpsim_drone.data import Settings, layout_split, load_dataset
from dvrpsim_drone.evaluator import DynamicEvaluator
from dvrpsim_drone.simulator import SIMULATOR_VERSION, DispatchConfig, run_episode
from gprt_vrptw.config import GPRTConfig
from gprt_vrptw.grammar import DRONE_FEATURES, Grammar, prefix_to_tree, tree_to_prefix, tree_to_string
from gprt_vrptw.splits import stratified_sample
from gprt_vrptw.trainer import GPRTTrainer
from run_gprt_experiment import run_gp_baseline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Dataset_DVRPTW_drone_GP_RNN"
INITIAL_TREES = ("trip_duration", "extra_duration", "extra_cost", "service_delay")
RULE_POOL = INITIAL_TREES + (
    ("sub", "extra_cost", "min_route_slack"),
    ("sub", "extra_cost", ("mul", 0.5, "min_route_slack")),
    ("sub", "extra_cost", ("mul", 2.0, "min_route_slack")),
    ("sub", "extra_duration", "min_route_slack"),
    ("sub", "extra_duration", ("mul", 0.5, "min_route_slack")),
    ("sub", "extra_duration", ("mul", 2.0, "min_route_slack")),
    ("add", "trip_duration", "extra_truck_km"),
    ("add", "extra_cost", "extra_duration"),
    ("add", "extra_cost", "service_wait"),
    "extra_truck_km", "slack",
)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def evaluate(tree, instances, settings, dispatch):
    rows = []
    for instance in instances:
        result = run_episode(instance, settings, dispatch, tree)
        latencies = sorted(result.decision_seconds)
        p95 = latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else 0
        rows.append({
            "instance": instance.name,
            "customers": instance.customer_count,
            "on_time": result.delivered,
            "on_time_rate": result.delivered / instance.customer_count,
            "unserved": instance.customer_count - result.delivered,
            "truck_km": result.truck_km,
            "drone_km": result.drone_km,
            "weighted_km": result.truck_km + dispatch.drone_cost_ratio * result.drone_km,
            "vehicles": result.vehicles,
            "drone_deliveries": result.drone_deliveries,
            "rendezvous_wait": result.rendezvous_wait,
            "reassignments": result.assignments_changed,
            "decision_p95_seconds": p95,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "train", "test"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fleet", type=int, required=True, help="Truck count per instance")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--methods", nargs="+", choices=("gp", "gprt"), default=["gp", "gprt"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "dvrpsim_drone_quality.json")
    parser.add_argument("--hold-minutes", type=float, default=5.0)
    parser.add_argument("--drone-cost-ratio", type=float, default=0.25)
    parser.add_argument("--truck-only", action="store_true")
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--depart-after-hold", action="store_true",
                        help="Depart after the hold even when the first customers are not ready")
    parser.add_argument("--retry-earlier-gaps", action="store_true",
                        help="Try earlier insertion positions when the usual positions all fail")
    parser.add_argument("--screen-initial-rules", action="store_true",
                        help="Seed both methods with rules selected on training instances")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Continue GPRT from last.pt")
    args = parser.parse_args()
    if args.output is None and args.stage != "inspect":
        parser.error("--output is required for train and test")
    if any(not math.isfinite(value) or value < 0
           for value in (args.hold_minutes, args.drone_cost_ratio)):
        parser.error("hold and drone cost ratio must be finite and nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error("seeds and methods must be unique")
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.resume and (args.stage != "train" or args.methods != ["gprt"]):
        parser.error("--resume requires train --methods gprt")
    import torch
    torch.set_num_threads(args.threads)

    settings = Settings.read(args.data / "dvrptw-setting.txt")
    dispatch = DispatchConfig(hold_minutes=args.hold_minutes, drone_enabled=not args.truck_only,
                              drone_cost_ratio=args.drone_cost_ratio,
                              wait_at_depot=not args.depart_after_hold,
                              retry_earlier_gaps=args.retry_earlier_gaps)
    instances = load_dataset(args.data, settings, args.fleet)
    split = layout_split(instances, args.fold)
    manifest = {
        "settings": asdict(settings), "dispatch": asdict(dispatch), "fleet": args.fleet,
        "fold": args.fold, "features": list(DRONE_FEATURES), "simulator": SIMULATOR_VERSION,
        "initial_trees": list(INITIAL_TREES),
        "batching": "benchmark-family-v2",
        "instances": {i.name: {"sha256": i.sha256, "layout": i.layout, "split": split[i.name]}
                      for i in instances},
    }
    if not dispatch.retry_earlier_gaps:
        del manifest["dispatch"]["retry_earlier_gaps"]
    if args.screen_initial_rules:
        manifest["initial_rule_pool"] = [tree_to_prefix(tree) for tree in RULE_POOL]
    if args.stage == "inspect":
        print(json.dumps(Counter(split.values()), sort_keys=True))
        return

    output = args.output
    manifest_path = output / "benchmark.json"
    if args.stage == "train":
        if args.resume and not manifest_path.is_file():
            parser.error("--resume requires an existing benchmark manifest")
        output.mkdir(parents=True, exist_ok=True)
        if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            parser.error("output directory belongs to another benchmark protocol")
        write_json(manifest_path, manifest)
        config = GPRTConfig(**json.loads(args.config.read_text(encoding="utf-8-sig")))
        config.validate()
        train = [item for item in instances if split[item.name] == "train"]
        validation = [item for item in instances if split[item.name] == "validation"]
        initial_trees = INITIAL_TREES
        rule_screening = None
        if args.screen_initial_rules:
            evaluator = DynamicEvaluator(settings, dispatch, config.complexity_penalty)
            grammar = Grammar(config.max_tokens, config.max_depth, features=DRONE_FEATURES)
            screen = stratified_sample(train, config.screening_train_size)
            ranked = sorted(
                (evaluator.fitness(tree, screen, phase="screening"), tree_to_string(tree), tree)
                for tree in RULE_POOL if grammar.validate_expression(tree_to_prefix(tree))
            )
            initial_trees = tuple(tree for _, _, tree in ranked[:len(INITIAL_TREES)])
            rule_screening = asdict(evaluator.stats)
            write_json(output / "initial_rule_screening.json", {
                "instances": [item.name for item in screen],
                "rules": [{"expression": name, "fitness": fitness._asdict()}
                          for fitness, name, _ in ranked],
                "selected": [tree_to_prefix(tree) for tree in initial_trees],
                "evaluation": rule_screening,
            })
        for seed in args.seeds:
            for method in args.methods:
                folder = output / method / str(seed)
                resume = folder / "last.pt" if args.resume else None
                if resume is None:
                    folder.mkdir(parents=True, exist_ok=False)
                elif not resume.is_file() or (folder / "policy.json").exists():
                    parser.error(f"Expected an unfinished GPRT run with a checkpoint: {folder}")
                evaluator = DynamicEvaluator(settings, dispatch, config.complexity_penalty)
                grammar = Grammar(config.max_tokens, config.max_depth, features=DRONE_FEATURES)
                if method == "gp":
                    tree, fitness, _ = run_gp_baseline(train, validation, config, seed,
                                                       evaluator=evaluator, grammar=grammar,
                                                       initial_trees=initial_trees, progress=True)
                else:
                    trainer = GPRTTrainer(config, evaluator=evaluator, grammar=grammar,
                                          initial_trees=initial_trees)
                    result = trainer.train(train, validation, seed=seed, output_dir=folder,
                                           split_manifest=split, resume=resume)
                    tree, fitness = result.best_tree, result.best_validation_fitness
                if tree is None or fitness is None:
                    raise RuntimeError("Validation did not select a heuristic")
                (folder / "final_heuristic.txt").write_text(tree_to_string(tree) + "\n", encoding="utf-8")
                write_json(folder / "policy.json", {
                    "protocol": identity(manifest), "method": method, "seed": seed,
                    "tokens": tree_to_prefix(tree), "validation_fitness": fitness._asdict(),
                    "training": asdict(evaluator.stats), "config": config.to_dict(),
                    "initial_rule_screening": rule_screening,
                })
        return

    if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        parser.error("test data and dispatch rules must match training")
    part = "transfer" if args.transfer else "test"
    test = [item for item in instances if split[item.name] == part]
    policies = []
    for seed in args.seeds:
        for method in args.methods:
            policy = json.loads((output / method / str(seed) / "policy.json").read_text(encoding="utf-8"))
            if (policy["protocol"], policy["method"], policy["seed"]) != (identity(manifest), method, seed):
                raise ValueError("Policy identity does not match requested run")
            config = policy["config"]
            grammar = Grammar(config["max_tokens"], config["max_depth"], features=DRONE_FEATURES)
            if not grammar.validate_expression(policy["tokens"]):
                raise ValueError("Policy does not match the dynamic grammar")
            policies.append((method, seed, prefix_to_tree(policy["tokens"])))
    results = [
        {"method": name, "seed": None, "rows": evaluate(tree, test, settings, dispatch)}
        for name, tree in (("greedy", None), ("edd", "slack"), ("trip_duration", "trip_duration"))
    ]
    for method, seed, tree in policies:
        results.append({"method": method, "seed": seed, "tokens": tree_to_prefix(tree),
                        "rows": evaluate(tree, test, settings, dispatch)})
    write_json(output / f"{part}_results.json", results)


if __name__ == "__main__":
    main()

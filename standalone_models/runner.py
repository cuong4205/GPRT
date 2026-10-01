from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from gprt_vrptw.grammar import CORE_FEATURES, tree_to_string
from gprt_vrptw.splits import split_solomon
from vrptw_baselines.data import VRPTWInstance, discover_instances

from .rl import RLConfig, evaluate_rl, train_rl
from .transformer import (
    TransformerConfig,
    evaluate_transformer,
    train_transformer,
)


def _solution_row(instance: VRPTWInstance, solution: Any) -> dict[str, Any]:
    return {
        "instance": instance.name,
        "feasible": solution.feasible,
        "vehicle_count": solution.vehicle_count,
        "distance": solution.distance,
        "waiting_time": solution.waiting_time,
        "unserved": list(solution.unserved),
        "routes": [list(route) for route in solution.routes],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    instances = discover_instances(args.data_dir)
    if not instances:
        raise RuntimeError(f"No VRPTW instances found under {args.data_dir}")
    split_map = split_solomon(instances)
    train = [item for item in instances if split_map[item.name.lower()] == "train"]
    validation = [item for item in instances if split_map[item.name.lower()] == "validation"]
    test = [item for item in instances if split_map[item.name.lower()] == "test"]
    if not train or not validation or not test:
        raise RuntimeError("The deterministic train/validation/test split is empty")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.model == "transformer":
        config = TransformerConfig(max_tokens=args.max_tokens, max_depth=args.max_depth)
        result = train_transformer(
            train,
            validation,
            seed=args.seed,
            cycles=args.cycles,
            samples_per_cycle=args.samples,
            batch_size_instances=args.batch_size,
            device=args.device,
            config=config,
        )
        test_fitness, test_solutions = evaluate_transformer(
            result.model, result.grammar, result.tree, test
        )
        validation_fitness = result.validation_fitness
        artifact = {
            "model": "transformer",
            "seed": args.seed,
            "config": asdict(config),
            "model_state": result.model.state_dict(),
            "vocabulary": result.grammar.vocab,
            "selected_tree": result.tree,
            "selected_expression": tree_to_string(result.tree),
            "validation_fitness": list(validation_fitness),
        }
        history = result.history
        expression = tree_to_string(result.tree)
    else:
        config = RLConfig()
        result = train_rl(
            train,
            validation,
            seed=args.seed,
            cycles=args.cycles,
            episodes_per_cycle=args.episodes,
            device=args.device,
            config=config,
        )
        test_fitness, test_solutions = evaluate_rl(result.model, test)
        validation_fitness = result.validation_fitness
        artifact = {
            "model": "rl",
            "seed": args.seed,
            "config": asdict(config),
            "feature_names": CORE_FEATURES,
            "model_state": result.model.state_dict(),
            "validation_fitness": list(validation_fitness),
        }
        history = result.history
        expression = None

    checkpoint_path = output_dir / f"{args.model}_seed{args.seed}.pt"
    torch.save(artifact, checkpoint_path)
    summary = {
        "model": args.model,
        "seed": args.seed,
        "split_counts": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "selected_expression": expression,
        "validation_fitness": list(validation_fitness),
        "test_fitness": list(test_fitness),
        "test_solutions": [
            _solution_row(instance, solution)
            for instance, solution in zip(test, test_solutions)
        ],
        "training_history": history,
        "checkpoint": str(checkpoint_path.resolve()),
    }
    results_path = output_dir / f"{args.model}_seed{args.seed}_results.json"
    results_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a standalone Transformer or RL VRPTW model."
    )
    parser.add_argument("--model", required=True, choices=("transformer", "rl"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="standalone_models/results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cycles", type=int, default=24)
    parser.add_argument("--samples", type=int, default=16, help="Transformer samples per cycle")
    parser.add_argument("--episodes", type=int, default=8, help="RL episodes per cycle")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()

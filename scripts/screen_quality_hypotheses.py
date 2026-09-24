"""Isolated train/validation screening; production model defaults stay unchanged.

Run one variant per process, e.g.:
python -m scripts.screen_quality_hypotheses --variant baseline --seed 20260914
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import platform
import time

import torch

from gprt_vrptw.config import GPRTConfig
from gprt_vrptw.evaluator import FitnessEvaluator, rank_rewards
from gprt_vrptw.grammar import tree_to_string
from gprt_vrptw.splits import split_solomon
import gprt_vrptw.trainer as trainer_module
from run_gprt_experiment import MemorySampler
from vrptw_baselines.data import discover_instances
from vrptw_baselines.solver import validate_solution


VARIANTS = ("baseline", "routing_rank", "elite3")


class ReplayEvaluator(FitnessEvaluator):
    """Reuse exact deterministic solutions while retaining per-run logical counts.

    Each run starts with an empty logical cache. A replayed first use is counted as
    a logical evaluation, with its physical work recorded separately as replay.
    """

    def __init__(self, *args, replay_cache=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_cache = replay_cache or {}
        self.replay_hits = 0

    def solution(self, tree, instance, *, phase="train"):
        key = (tree_to_string(tree), instance.sha256, self.config_hash)
        replayed = key not in self.cache and key in self.replay_cache
        if replayed:
            self.cache[key] = self.replay_cache[key]
        result = super().solution(tree, instance, phase=phase)
        if replayed:
            self.replay_hits += 1
            self.stats.cache_hits -= 1
            self.stats.unique_evaluations += 1
            self.stats.instance_evaluations += 1
            field = f"{phase}_instance_evaluations"
            setattr(self.stats, field, getattr(self.stats, field) + 1)
        return result


def reward_diagnostic(fitnesses):
    count = len(fitnesses)
    total_pairs = count * (count - 1) // 2
    full = Counter(fitnesses)
    routing = Counter(tuple(f[:4]) for f in fitnesses)
    full_ties = sum(n * (n - 1) // 2 for n in full.values())
    routing_ties = sum(n * (n - 1) // 2 for n in routing.values())
    return {
        "samples": count,
        "full_tied_pairs": full_ties,
        "routing_tied_pairs": routing_ties,
        "pairs_separated_only_by_complexity": routing_ties - full_ties,
        "all_pairs": total_pairs,
        "complexity_only_pair_fraction": (routing_ties - full_ties) / max(1, total_pairs),
    }


def run(variant: str, seed: int, root: Path, replay_from: list[Path] | None = None) -> dict:
    output = root / variant / str(seed)
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing experiment: {output}")
    output.mkdir(parents=True)
    instances = discover_instances(Path("data/data/Solomon"))
    split = split_solomon(instances)
    train = [i for i in instances if split[i.name.lower()] == "train"]
    validation = [i for i in instances if split[i.name.lower()] == "validation"]
    assert len(train) == 32 and len(validation) == 12
    config = GPRTConfig(cycles=8)
    if variant == "elite3":
        config = replace(config, elite_fraction=3 / config.population_size)
    metadata = {
        "variant": variant, "seed": seed, "config": config.to_dict(),
        "test_evaluated": False, "torch": torch.__version__,
        "python": platform.python_version(), "torch_threads": torch.get_num_threads(),
        "code_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for folder in ("gprt_vrptw", "vrptw_baselines")
                        for p in sorted(Path(folder).glob("*.py"))},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "split": split,
        "instance_sha256": {i.name: i.sha256 for i in instances},
        "replay_from": [str(p) for p in (replay_from or [])],
        "counting": "Per-run logical evaluations; physical replay hits reported separately. Runtime with replay is not a cold runtime comparison.",
    }
    replay_cache = {}
    metadata["replay_source_details"] = []
    for source in replay_from or []:
        source_metadata = json.loads((source / "audit_metadata.json").read_text(encoding="utf-8"))
        if source_metadata["code_sha256"] != metadata["code_sha256"]:
            raise ValueError(f"Replay source uses different model/evaluator code: {source}")
        if source_metadata["instance_sha256"] != metadata["instance_sha256"]:
            raise ValueError(f"Replay source uses different data: {source}")
        checkpoint_bytes = (source / "last.pt").read_bytes()
        state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
        metadata["replay_source_details"].append({
            "source": str(source), "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "completed_cycles_at_snapshot": state["training_step"],
            "note": "Only cached deterministic solutions are reused; model, archive, optimizer and RNG are not reused.",
        })
        replay_cache.update(state["evaluator_cache"])
        del state, checkpoint_bytes
    (output / "audit_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    diagnostics = []
    original_rank = trainer_module.rank_rewards
    original_evaluator = trainer_module.FitnessEvaluator

    def observed_rank(fitnesses):
        diagnostics.append(reward_diagnostic(fitnesses))
        if variant == "routing_rank":
            return rank_rewards([f._replace(complexity=0.0) for f in fitnesses])
        return original_rank(fitnesses)

    trainer_module.rank_rewards = observed_rank
    if replay_cache:
        trainer_module.FitnessEvaluator = lambda *a, **kw: ReplayEvaluator(*a, replay_cache=replay_cache, **kw)
    started = time.perf_counter()
    memory = MemorySampler()
    memory.start()
    print(json.dumps({"event": "start", "variant": variant, "seed": seed}), flush=True)
    try:
        trainer = trainer_module.GPRTTrainer(config)
        with (output / "training.jsonl").open("w", encoding="utf-8") as handle:
            with redirect_stdout(handle):
                result = trainer.train(train, validation, seed=seed, output_dir=output, split_manifest=split)
        elapsed = time.perf_counter() - started
        training_stats = asdict(trainer.evaluator.stats)
        assert result.best_tree is not None
        validation_rows = []
        for instance in validation:
            solution = trainer.evaluator.solution(result.best_tree, instance, phase="validation")
            checked = validate_solution(instance, solution.routes)
            assert checked["feasible"] == solution.feasible
            assert abs(float(checked["distance"]) - solution.distance) < 1e-8
            validation_rows.append({
                "instance": instance.name, "feasible": solution.feasible,
                "unserved": len(solution.unserved), "vehicles": solution.vehicle_count,
                "distance": solution.distance,
            })
        summary = {
            "variant": variant, "seed": seed, "test_evaluated": False,
            "validation_fitness": list(result.best_validation_fitness),
            "expression": tree_to_string(result.best_tree),
            "end_to_end_train_seconds": elapsed,
            "peak_rss_mb": memory.peak_bytes / 1024**2,
            "training_stats": training_stats,
            "replayed_logical_evaluations": getattr(trainer.evaluator, "replay_hits", 0),
            "physical_instance_evaluations": training_stats["instance_evaluations"] - getattr(trainer.evaluator, "replay_hits", 0),
            "training_coverage": trainer.logs[-1]["training_coverage"],
            "gradient_steps": result.completed_cycles,
            "reward_diagnostics": diagnostics,
            "validation_rows": validation_rows,
        }
        (output / "quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"event": "complete", **{k: summary[k] for k in (
            "variant", "seed", "validation_fitness", "expression", "end_to_end_train_seconds", "peak_rss_mb"
        )}}), flush=True)
        return summary
    finally:
        memory.stop()
        trainer_module.rank_rewards = original_rank
        trainer_module.FitnessEvaluator = original_evaluator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, choices=(20260914, 20260915), required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/gprt_vrptw/quality_screening_20260917"))
    parser.add_argument("--replay-from", nargs="*", type=Path)
    args = parser.parse_args()
    run(args.variant, args.seed, args.output, args.replay_from)

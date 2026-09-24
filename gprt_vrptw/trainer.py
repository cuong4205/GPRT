from __future__ import annotations

import csv
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from vrptw_baselines.data import VRPTWInstance

from .config import GPRTConfig
from .evaluator import (
    EvaluationStats,
    Fitness,
    FitnessEvaluator,
    behavior_fingerprint,
    rank_rewards,
)
from .gp import evolve_population
from .grammar import Grammar, Tree, tree_depth, tree_size, tree_to_string
from .selection import record_observation, screen_archive, select_archive_seeds
from .splits import stratified_batches, stratified_sample
from .transformer import CausalTransformer


@dataclass(slots=True)
class TrainingResult:
    seed: int
    completed_cycles: int
    best_validation_fitness: Fitness | None
    best_tree: Tree | None
    best_checkpoint: Path | None
    last_checkpoint: Path
    logs: list[dict[str, Any]]
    archive: dict[str, dict[str, Any]]
    training_instance_names: tuple[str, ...]
    validation_instance_names: tuple[str, ...]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _fitness_json(fitness: Fitness | None) -> list[float] | None:
    return None if fitness is None else list(fitness)


def _stats_delta(before: EvaluationStats, after: EvaluationStats) -> dict[str, float | int]:
    return {field: getattr(after, field) - getattr(before, field) for field in asdict(before)}


def _branch_gradient_metrics(
    policy_loss: torch.Tensor,
    imitation_loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> tuple[float, float, float]:
    """Norms and cosine before clipping; zero is reported for inactive branches."""
    vectors: list[torch.Tensor] = []
    for loss in (policy_loss, imitation_loss):
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True
        )
        vectors.append(
            torch.cat(
                [
                    (gradient if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
                    for gradient, parameter in zip(gradients, parameters)
                ]
            )
        )
    policy_vector, imitation_vector = vectors
    policy_norm = float(torch.linalg.vector_norm(policy_vector).item())
    imitation_norm = float(torch.linalg.vector_norm(imitation_vector).item())
    cosine = 0.0
    if policy_norm > 0 and imitation_norm > 0:
        cosine = float(
            torch.dot(policy_vector, imitation_vector).item()
            / (policy_norm * imitation_norm)
        )
    return policy_norm, imitation_norm, cosine


def normalized_advantages(
    rewards: torch.Tensor,
    baseline: float,
    mode: str,
) -> torch.Tensor:
    advantages = (rewards - baseline).detach()
    scale = float(rewards.std(unbiased=False).item())
    if mode == "ewma_scale" and scale > 0:
        return advantages / scale
    if mode == "batch_standardize" and scale > 0:
        return (advantages - advantages.mean()) / scale
    return advantages


def _behaviorally_diverse_elites(
    scored_trees: list[tuple[Fitness, Tree]],
    count: int,
    probe: list[VRPTWInstance],
    evaluator: FitnessEvaluator,
) -> list[Tree]:
    selected: list[Tree] = []
    fingerprints: set[tuple[tuple[tuple[int, ...], ...], ...]] = set()
    for _, tree in scored_trees:
        fingerprint = behavior_fingerprint(tree, probe, evaluator)
        if fingerprint not in fingerprints:
            selected.append(tree)
            fingerprints.add(fingerprint)
        if len(selected) == count:
            return selected
    for _, tree in scored_trees:
        if tree not in selected:
            selected.append(tree)
        if len(selected) == count:
            break
    return selected


class GPRTTrainer:
    def __init__(self, config: GPRTConfig, *, device: str = "cpu"):
        config.validate()
        self.config = config
        self.grammar = Grammar(
            config.max_tokens,
            config.max_depth,
            typed=config.typed_grammar,
            include_lookahead=config.use_lookahead_features,
        )
        self.device = torch.device(device)
        self.model = CausalTransformer(self.grammar, config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.evaluator = FitnessEvaluator(
            config.complexity_penalty,
            use_lookahead_features=config.use_lookahead_features,
        )
        self.baseline: float | None = None
        self.archive: dict[str, dict[str, Any]] = {}
        self.logs: list[dict[str, Any]] = []
        self.best_validation_fitness: Fitness | None = None
        self.best_tree: Tree | None = None
        self.completed_cycles = 0
        self.seen_training_instances: set[str] = set()
        self.screening_instance_names: tuple[str, ...] = ()
        self.rng = random.Random()

    def _validate(
        self,
        screening: list[VRPTWInstance],
        validation: list[VRPTWInstance],
    ) -> tuple[Fitness, Tree]:
        candidates = screen_archive(
            self.archive,
            self.evaluator,
            screening,
            self.config.validation_candidates,
        )
        if not candidates:
            raise RuntimeError("Candidate archive is empty at validation")
        evaluated = [
            (
                self.evaluator.fitness(
                    item["tree"], validation, phase="validation"
                ),
                item["tree"],
            )
            for _, item in candidates
        ]
        return min(evaluated, key=lambda item: item[0])

    def _state(self, seed: int, split_manifest: dict[str, str]) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "vocabulary": list(self.grammar.vocab),
            "grammar": {
                "max_tokens": self.grammar.max_tokens,
                "max_depth": self.grammar.max_depth,
                "typed": self.grammar.typed,
            },
            "split": split_manifest,
            "archive": self.archive,
            "logs": self.logs,
            "training_step": self.completed_cycles,
            "baseline": self.baseline,
            "seen_training_instances": sorted(self.seen_training_instances),
            "screening_instance_names": self.screening_instance_names,
            "best_validation_fitness": self.best_validation_fitness,
            "best_validation_heuristic": self.best_tree,
            "seed": seed,
            "rng_state": {
                "python_global": random.getstate(),
                "python_local": self.rng.getstate(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "evaluator_cache": self.evaluator.cache,
            "evaluator_stats": self.evaluator.stats,
        }

    def save_checkpoint(self, path: Path, seed: int, split_manifest: dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._state(seed, split_manifest), path)

    def load_checkpoint(self, path: Path) -> int:
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state["config"] != self.config.to_dict():
            raise ValueError("Resume config differs from checkpoint config")
        if state["vocabulary"] != list(self.grammar.vocab):
            raise ValueError("Checkpoint vocabulary differs from current grammar")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.archive = state["archive"]
        self.logs = state["logs"]
        self.completed_cycles = int(state["training_step"])
        self.baseline = state["baseline"]
        self.seen_training_instances = set(state.get("seen_training_instances", []))
        self.screening_instance_names = tuple(state.get("screening_instance_names", ()))
        self.best_validation_fitness = state["best_validation_fitness"]
        self.best_tree = state["best_validation_heuristic"]
        self.evaluator.cache = state.get("evaluator_cache", {})
        self.evaluator.stats = state.get("evaluator_stats", EvaluationStats())
        random.setstate(state["rng_state"]["python_global"])
        self.rng.setstate(state["rng_state"]["python_local"])
        torch.set_rng_state(state["rng_state"]["torch_cpu"])
        if torch.cuda.is_available() and state["rng_state"]["torch_cuda"] is not None:
            torch.cuda.set_rng_state_all(state["rng_state"]["torch_cuda"])
        return int(state["seed"])

    def train(
        self,
        train_instances: list[VRPTWInstance],
        validation_instances: list[VRPTWInstance],
        *,
        seed: int,
        output_dir: Path,
        split_manifest: dict[str, str],
        resume: Path | None = None,
        stop_after_cycle: int | None = None,
    ) -> TrainingResult:
        if not train_instances or not validation_instances:
            raise ValueError("Non-empty train and validation sets are required")
        output_dir.mkdir(parents=True, exist_ok=True)
        if resume is None:
            seed_everything(seed)
            # Model construction happened before the run seed was known; rebuild so
            # the public training seed covers initialization as well as sampling.
            self.model = CausalTransformer(self.grammar, self.config).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            self.rng.seed(seed)
        else:
            checkpoint_seed = self.load_checkpoint(resume)
            if checkpoint_seed != seed:
                raise ValueError("Resume seed differs from checkpoint")
        batches = stratified_batches(
            train_instances, self.config.batch_size_instances, seed=seed
        )
        if self.config.require_full_train_coverage and self.config.cycles < len(batches):
            raise ValueError(
                f"cycles={self.config.cycles} covers only {self.config.cycles}/"
                f"{len(batches)} training batches; use at least {len(batches)} cycles"
            )
        screening_instances = stratified_sample(
            train_instances, self.config.screening_train_size
        )
        expected_screening_names = tuple(
            instance.name.lower() for instance in screening_instances
        )
        if self.screening_instance_names and self.screening_instance_names != expected_screening_names:
            raise ValueError("Resume screening set differs from checkpoint")
        self.screening_instance_names = expected_screening_names
        behavior_probe = screening_instances[: self.config.behavior_probe_size]
        last_checkpoint = output_dir / "last.pt"
        best_checkpoint = output_dir / "best.pt"
        limit = self.config.cycles if stop_after_cycle is None else min(stop_after_cycle, self.config.cycles)

        for cycle in range(self.completed_cycles + 1, limit + 1):
            wall_started = time.perf_counter()
            stats_before = EvaluationStats(**asdict(self.evaluator.stats))
            batch = batches[(cycle - 1) % len(batches)]
            self.seen_training_instances.update(instance.name.lower() for instance in batch)
            self.model.train()
            transformer_started = time.perf_counter()
            if self.config.use_gp:
                samples = self.model.generate(self.config.transformer_samples, greedy=False)
            else:
                # Equal-fitness-budget standalone RL baseline. Sampling and the
                # later likelihood pass both use eval mode, so recomputed
                # on-policy log-probabilities are not perturbed by dropout masks.
                self.model.eval()
                with torch.no_grad():
                    samples = self.model.generate(
                        self.config.standalone_rl_samples_per_cycle, greedy=False
                    )
            transformer_sample_seconds = time.perf_counter() - transformer_started
            on_policy_trees = [sample.tree for sample in samples]

            neural_expressions = {tree_to_string(tree) for tree in on_policy_trees}
            if self.config.use_gp:
                neural_seed_trees = (
                    list(on_policy_trees) if self.config.use_transformer_seeding else []
                )
                archive_seed_trees, archive_seed_pool_size = select_archive_seeds(
                    self.archive,
                    self.evaluator,
                    screening_instances,
                    behavior_probe,
                    self.config.archive_seed_count,
                    excluded_expressions=neural_expressions,
                )
                seeds = neural_seed_trees + archive_seed_trees
                random_seed_trees: list[Tree] = []
                while len(seeds) < self.config.population_size:
                    tree = self.grammar.random_tree(self.rng, grow=len(seeds) % 2 == 0)
                    seeds.append(tree)
                    random_seed_trees.append(tree)
            else:
                neural_seed_trees = []
                archive_seed_trees = []
                archive_seed_pool_size = 0
                random_seed_trees = []
                seeds = []

            gp_started = time.perf_counter()
            if self.config.use_gp:
                final_population, gp_scores, generation_best = evolve_population(
                    seeds, batch, self.evaluator, self.rng, self.grammar, self.config
                )
            else:
                # Transformer+RL ablation: no random/GP individual may affect the
                # policy reward or candidate archive.
                final_population = list(on_policy_trees)
                gp_scores = {}
                generation_best = []
            gp_seconds = time.perf_counter() - gp_started

            initial_fitness = [self.evaluator.fitness(tree, batch) for tree in on_policy_trees]
            final_fitness = (
                [self.evaluator.fitness(tree, batch) for tree in final_population]
                if self.config.use_gp
                else initial_fitness
            )
            for tree, fitness in zip(on_policy_trees, initial_fitness):
                record_observation(self.archive, tree, fitness, cycle, batch)
            for tree, fitness in zip(final_population, final_fitness):
                record_observation(self.archive, tree, fitness, cycle, batch)

            # Only current-policy samples define the REINFORCE rank pool. GP
            # individuals are off-policy and cannot move a neural sample's reward.
            all_rewards = rank_rewards(initial_fitness)
            rewards = torch.tensor(
                all_rewards[: len(samples)], dtype=torch.float32, device=self.device
            ).detach()
            reward_mean = float(rewards.mean().item())
            reward_std = float(rewards.std(unbiased=False).item())
            baseline_for_loss = (
                reward_mean
                if self.config.advantage_mode == "batch_standardize" or self.baseline is None
                else self.baseline
            )
            advantages = normalized_advantages(
                rewards, baseline_for_loss, self.config.advantage_mode
            )
            if self.config.use_gp:
                log_probs = torch.stack([sample.log_probability for sample in samples])
                sample_entropy = torch.stack([sample.entropy for sample in samples]).mean()
            else:
                log_probs, _, sample_entropy = self.model.teacher_forced(on_policy_trees)
            policy_loss = -(advantages * log_probs).mean() if self.config.use_reinforce else log_probs.sum() * 0.0

            elite_count = max(1, math.ceil(len(final_population) * self.config.elite_fraction))
            batch_elite_pairs = sorted(
                zip(final_fitness, final_population), key=lambda item: item[0]
            )[: self.config.elite_screening_candidates]
            # Avoid imitating a tree merely because it met an easy current batch.
            if self.config.use_elite_imitation:
                screened_elites = sorted(
                    [
                        (
                        self.evaluator.fitness(
                            tree, screening_instances, phase="screening"
                        ),
                        tree,
                        )
                        for _, tree in batch_elite_pairs
                    ],
                    key=lambda item: (item[0], tree_to_string(item[1])),
                )
                elite_trees = _behaviorally_diverse_elites(
                    screened_elites, elite_count, screening_instances, self.evaluator
                )
            else:
                elite_trees = [tree for _, tree in batch_elite_pairs[:elite_count]]
            transformer_train_started = time.perf_counter()
            if self.config.use_elite_imitation:
                _, elite_nll, teacher_entropy = self.model.teacher_forced(elite_trees)
            else:
                elite_nll = self.model.output.weight.sum() * 0.0
                teacher_entropy = sample_entropy
            entropy = 0.5 * (sample_entropy + teacher_entropy)
            weighted_elite_loss = self.config.alpha * elite_nll
            loss = policy_loss + weighted_elite_loss - self.config.beta * entropy
            parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            policy_gradient_norm, elite_gradient_norm, gradient_cosine = _branch_gradient_metrics(
                policy_loss, weighted_elite_loss, parameters
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip))
            self.optimizer.step()
            transformer_train_seconds = time.perf_counter() - transformer_train_started
            self.baseline = (
                reward_mean
                if self.baseline is None
                else self.config.baseline_decay * self.baseline + (1 - self.config.baseline_decay) * reward_mean
            )

            validation_fitness: Fitness | None = None
            did_validate = cycle % self.config.validation_interval == 0 or cycle == self.config.cycles
            if did_validate:
                validation_fitness, validation_tree = self._validate(
                    screening_instances, validation_instances
                )
                if self.best_validation_fitness is None or validation_fitness < self.best_validation_fitness:
                    self.best_validation_fitness = validation_fitness
                    self.best_tree = validation_tree

            self.completed_cycles = cycle
            current_best = min(final_fitness)
            population_expressions = [tree_to_string(tree) for tree in final_population]
            behavior_fingerprints = [
                behavior_fingerprint(tree, behavior_probe, self.evaluator)
                for tree in final_population
            ]
            random_fitness = [
                self.evaluator.fitness(tree, batch, phase="diagnostic")
                for tree in random_seed_trees
            ]
            random_median = (
                sorted(random_fitness)[len(random_fitness) // 2] if random_fitness else None
            )
            neural_better_than_random = (
                sum(fitness < random_median for fitness in initial_fitness) / len(initial_fitness)
                if random_median is not None and initial_fitness else None
            )
            final_elite_expressions = {
                tree_to_string(tree) for tree in elite_trees
            }
            archive_seed_fitness = [
                list(gp_scores[tree_to_string(tree)])
                for tree in archive_seed_trees
                if tree_to_string(tree) in gp_scores
            ]
            reward_counts = Counter(initial_fitness)
            tied_pairs = sum(count * (count - 1) // 2 for count in reward_counts.values())
            total_pairs = len(initial_fitness) * (len(initial_fitness) - 1) // 2
            stats = _stats_delta(stats_before, self.evaluator.stats)
            log = {
                "cycle": cycle,
                "batch_instances": [instance.name.upper() for instance in batch],
                "loss": float(loss.detach().item()),
                "policy_loss": float(policy_loss.detach().item()),
                "elite_nll": float(elite_nll.detach().item()),
                "entropy": float(entropy.detach().item()),
                "reward_mean": reward_mean,
                "reward_std": reward_std,
                "neural_reward_tie_rate": tied_pairs / max(1, total_pairs),
                "neural_mean_vehicles": sum(f.mean_vehicles for f in initial_fitness) / len(initial_fitness),
                "neural_mean_normalized_distance": (
                    sum(f.mean_normalized_distance for f in initial_fitness) / len(initial_fitness)
                ),
                "baseline_ewma": float(self.baseline),
                "advantage_mode": self.config.advantage_mode,
                "baseline_for_loss": float(baseline_for_loss),
                "advantage_mean": float(advantages.mean().item()),
                "advantage_std": float(advantages.std(unbiased=False).item()),
                "train_fitness": list(current_best),
                "validation_fitness": _fitness_json(validation_fitness),
                "feasibility_rate": 1.0 - current_best.infeasible_instances / len(batch),
                "mean_vehicles": current_best.mean_vehicles,
                "mean_normalized_distance": current_best.mean_normalized_distance,
                "best_expression": tree_to_string(final_population[final_fitness.index(current_best)]),
                "expression_length": tree_size(final_population[final_fitness.index(current_best)]),
                "expression_depth": tree_depth(final_population[final_fitness.index(current_best)]),
                "population_diversity": len(set(population_expressions)) / len(population_expressions),
                "behavior_diversity": len(set(behavior_fingerprints)) / len(behavior_fingerprints),
                "seen_training_instances": len(self.seen_training_instances),
                "training_coverage": len(self.seen_training_instances) / len(train_instances),
                "gradient_step": cycle,
                "neural_unique_samples": len(neural_expressions),
                "neural_seed_count": len(neural_seed_trees),
                "archive_seed_pool_size": archive_seed_pool_size,
                "archive_seed_count": len(archive_seed_trees),
                "archive_seed_fitness": archive_seed_fitness,
                "random_seed_count": len(random_seed_trees),
                "initial_seed_count": len(seeds),
                "neural_direct_survival_rate": (
                    sum(expression in neural_expressions for expression in population_expressions)
                    / len(population_expressions)
                ),
                "neural_better_than_random_rate": neural_better_than_random,
                "neural_direct_elite_count": len(final_elite_expressions & neural_expressions),
                "fitness_requests": stats["fitness_requests"],
                "unique_evaluations": stats["unique_evaluations"],
                "instance_evaluations": stats["instance_evaluations"],
                "train_instance_requests": stats["train_instance_requests"],
                "screening_instance_requests": stats["screening_instance_requests"],
                "validation_instance_requests": stats["validation_instance_requests"],
                "diagnostic_instance_requests": stats["diagnostic_instance_requests"],
                "train_instance_evaluations": stats["train_instance_evaluations"],
                "screening_instance_evaluations": stats["screening_instance_evaluations"],
                "validation_instance_evaluations": stats["validation_instance_evaluations"],
                "diagnostic_instance_evaluations": stats["diagnostic_instance_evaluations"],
                "cache_hits": stats["cache_hits"],
                "transformer_sample_seconds": transformer_sample_seconds,
                "transformer_train_seconds": transformer_train_seconds,
                "gp_seconds": gp_seconds,
                "evaluator_seconds": stats["evaluator_seconds"],
                "wall_seconds": time.perf_counter() - wall_started,
                "gradient_norm": gradient_norm,
                "policy_gradient_norm": policy_gradient_norm,
                "elite_gradient_norm": elite_gradient_norm,
                "gradient_cosine": gradient_cosine,
                "gp_generation_best": [list(value) for value in generation_best],
            }
            self.logs.append(log)
            self.save_checkpoint(last_checkpoint, seed, split_manifest)
            if did_validate and self.best_tree is not None and validation_fitness == self.best_validation_fitness:
                self.save_checkpoint(best_checkpoint, seed, split_manifest)
            print(json.dumps(log, ensure_ascii=False), flush=True)

        self._write_outputs(output_dir, seed, split_manifest)
        return TrainingResult(
            seed=seed,
            completed_cycles=self.completed_cycles,
            best_validation_fitness=self.best_validation_fitness,
            best_tree=self.best_tree,
            best_checkpoint=best_checkpoint if best_checkpoint.exists() else None,
            last_checkpoint=last_checkpoint,
            logs=self.logs,
            archive=self.archive,
            training_instance_names=tuple(i.name.lower() for i in train_instances),
            validation_instance_names=tuple(i.name.lower() for i in validation_instances),
        )

    def _write_outputs(self, output_dir: Path, seed: int, split_manifest: dict[str, str]) -> None:
        (output_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )
        (output_dir / "vocabulary.json").write_text(
            json.dumps(list(self.grammar.vocab), indent=2), encoding="utf-8"
        )
        (output_dir / "split_manifest.json").write_text(
            json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        serial_archive = []
        for expression, item in sorted(self.archive.items()):
            observations = [
                {
                    "batch_id": batch_id,
                    "fitness": list(observation["fitness"]),
                    "cycle": observation["cycle"],
                    "instances": list(observation["instances"]),
                }
                for batch_id, observation in sorted(item["observations"].items())
            ]
            serial_archive.append(
                {
                    "expression": expression,
                    "tokens": item["tokens"],
                    "observations": observations,
                    "screening_fitness": _fitness_json(item.get("screening_fitness")),
                    "screening_batch_id": item.get("screening_batch_id"),
                }
            )
        (output_dir / "candidate_archive.json").write_text(
            json.dumps(serial_archive, indent=2), encoding="utf-8"
        )
        summary = {
            "seed": seed,
            "completed_cycles": self.completed_cycles,
            "best_validation_fitness": _fitness_json(self.best_validation_fitness),
            "best_expression": tree_to_string(self.best_tree) if self.best_tree is not None else None,
            "seen_training_instances": sorted(self.seen_training_instances),
            "screening_instance_names": list(self.screening_instance_names),
            "training_coverage_count": len(self.seen_training_instances),
            "training_coverage_rate": (
                self.logs[-1]["training_coverage"] if self.logs else 0.0
            ),
            "gradient_steps": self.completed_cycles,
        }
        (output_dir / "training_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        if self.best_tree is not None:
            (output_dir / "final_heuristic.txt").write_text(
                tree_to_string(self.best_tree) + "\n", encoding="utf-8"
            )
        if self.logs:
            with (output_dir / "learning_curves.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.logs[0].keys())
                writer.writeheader()
                for row in self.logs:
                    writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()})

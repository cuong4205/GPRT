from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch
from torch import Tensor

from gprt_vrptw.evaluator import Fitness, FitnessEvaluator, make_priority, rank_rewards
from gprt_vrptw.grammar import Grammar, Tree, tree_to_string
from gprt_vrptw.splits import stratified_sample
from gprt_vrptw.transformer import CausalTransformer
from vrptw_baselines.data import VRPTWInstance
from vrptw_baselines.solver import Solution, solve

from .common import fitness_from_solutions, seed_everything


@dataclass(slots=True)
class TransformerConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    max_tokens: int = 31
    max_depth: int = 4
    temperature: float = 1.0
    learning_rate: float = 1e-3
    entropy_coefficient: float = 1e-3
    complexity_penalty: float = 1e-4


@dataclass(slots=True)
class TransformerRun:
    model: CausalTransformer
    grammar: Grammar
    tree: Tree
    validation_fitness: Fitness
    history: list[dict[str, float]]


def _batch(
    instances: list[VRPTWInstance], size: int, rng: random.Random
) -> list[VRPTWInstance]:
    if not instances:
        raise ValueError("Training instances must not be empty")
    return rng.sample(instances, min(size, len(instances)))


def train_transformer(
    train_instances: list[VRPTWInstance],
    validation_instances: list[VRPTWInstance],
    *,
    seed: int = 0,
    cycles: int = 24,
    samples_per_cycle: int = 16,
    batch_size_instances: int = 4,
    validation_candidates: int = 16,
    device: str = "cpu",
    config: TransformerConfig | None = None,
) -> TransformerRun:
    """Train a grammar-constrained Transformer policy with REINFORCE only."""
    if not train_instances or not validation_instances:
        raise ValueError("Non-empty train and validation sets are required")
    if cycles < 1 or samples_per_cycle < 2 or batch_size_instances < 1:
        raise ValueError("cycles, batch size, and at least two samples are required")
    config = config or TransformerConfig()
    if config.d_model % config.nhead:
        raise ValueError("d_model must be divisible by nhead")

    seed_everything(seed)
    rng = random.Random(seed)
    torch_device = torch.device(device)
    grammar = Grammar(config.max_tokens, config.max_depth, include_lookahead=False)
    model = CausalTransformer(grammar, config).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    evaluator = FitnessEvaluator(
        config.complexity_penalty, use_lookahead_features=False
    )
    archive: dict[str, tuple[Tree, Fitness]] = {}
    screening_instances = stratified_sample(train_instances, min(4, len(train_instances)))
    best_tree: Tree | None = None
    best_fitness: Fitness | None = None
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []
    validation_interval = max(1, cycles // 4)

    for cycle in range(1, cycles + 1):
        batch = _batch(train_instances, batch_size_instances, rng)
        model.train()
        samples = model.generate(samples_per_cycle)
        fitnesses = [
            evaluator.fitness(sample.tree, batch, phase="train") for sample in samples
        ]
        rewards = torch.tensor(
            rank_rewards(fitnesses), dtype=torch.float32, device=torch_device
        ).detach()
        log_probabilities = torch.stack([sample.log_probability for sample in samples])
        entropy = torch.stack([sample.entropy for sample in samples]).mean()
        policy_loss = -(rewards * log_probabilities).mean()
        loss = policy_loss - config.entropy_coefficient * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for sample, fitness in zip(samples, fitnesses):
            key = tree_to_string(sample.tree)
            previous = archive.get(key)
            if previous is None or fitness < previous[1]:
                archive[key] = (sample.tree, fitness)

        record = {
            "cycle": float(cycle),
            "policy_loss": float(policy_loss.detach().cpu()),
            "mean_reward": float(rewards.mean().cpu()),
            "entropy": float(entropy.detach().cpu()),
        }
        if cycle % validation_interval == 0 or cycle == cycles:
            screened = [
                (
                    evaluator.fitness(tree, screening_instances, phase="screening"),
                    tree,
                )
                for tree, _ in archive.values()
            ]
            finalists = sorted(
                ((tree, fitness) for fitness, tree in screened),
                key=lambda item: item[1],
            )[: max(1, validation_candidates)]
            validated = [
                (evaluator.fitness(tree, validation_instances, phase="validation"), tree)
                for tree, _ in finalists
            ]
            cycle_fitness, cycle_tree = min(validated, key=lambda item: item[0])
            record["validation_infeasible"] = float(cycle_fitness.infeasible_instances)
            record["validation_vehicles"] = float(cycle_fitness.mean_vehicles)
            record["validation_distance"] = float(cycle_fitness.mean_normalized_distance)
            if best_fitness is None or cycle_fitness < best_fitness:
                best_fitness, best_tree = cycle_fitness, cycle_tree
                best_state = copy.deepcopy(model.state_dict())
        history.append(record)

    if best_tree is None or best_fitness is None or best_state is None:
        raise RuntimeError("Transformer training did not produce a validated expression")
    model.load_state_dict(best_state)
    return TransformerRun(model, grammar, best_tree, best_fitness, history)


def evaluate_transformer(
    model: CausalTransformer,
    grammar: Grammar,
    tree: Tree,
    instances: list[VRPTWInstance],
) -> tuple[Fitness, list[Solution]]:
    del model, grammar  # The selected expression is the learned inference artifact.
    priority = make_priority(tree, use_lookahead_features=False)
    solutions = [
        solve(instance, "transformer", priority_function=priority)
        for instance in instances
    ]
    return fitness_from_solutions(instances, solutions), solutions

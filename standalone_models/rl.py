from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from gprt_vrptw.evaluator import Fitness, distance_scale, normalized_features
from gprt_vrptw.grammar import CORE_FEATURES
from vrptw_baselines.data import VRPTWInstance
from vrptw_baselines.solver import CandidateContext, Solution, solve

from .common import fitness_from_solutions, seed_everything


@dataclass(slots=True)
class RLConfig:
    hidden_size: int = 64
    learning_rate: float = 1e-3
    entropy_coefficient: float = 1e-3


@dataclass(slots=True)
class RLRun:
    model: CandidateActor
    validation_fitness: Fitness
    history: list[dict[str, float]]


class CandidateActor(nn.Module):
    """Score feasible next-customer actions from shared normalized features."""

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(CORE_FEATURES), hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


class _ActorSelector:
    def __init__(self, model: CandidateActor, *, stochastic: bool):
        self.model = model
        self.stochastic = stochastic
        self.log_probabilities: list[Tensor] = []
        self.entropies: list[Tensor] = []

    def __call__(
        self, instance: VRPTWInstance, contexts: list[CandidateContext]
    ) -> CandidateContext:
        vectors = [
            tuple(
                normalized_features(instance, context, use_lookahead_features=False)[name]
                for name in CORE_FEATURES
            )
            for context in contexts
        ]
        inputs = torch.tensor(
            vectors, dtype=torch.float32, device=next(self.model.parameters()).device
        )
        logits = self.model(inputs)
        distribution = torch.distributions.Categorical(logits=logits)
        if self.stochastic:
            action = distribution.sample()
            self.log_probabilities.append(distribution.log_prob(action))
            self.entropies.append(distribution.entropy())
        else:
            action = torch.argmax(logits)
        return contexts[int(action.item())]


def _episode_reward(instance: VRPTWInstance, solution: Solution) -> float:
    unserved_fraction = len(solution.unserved) / max(1, instance.customer_count)
    normalized_distance = solution.distance / (
        max(1, instance.customer_count) * distance_scale(instance)
    )
    return (
        -1000.0 * float(not solution.feasible)
        -100.0 * unserved_fraction
        -float(solution.vehicle_count)
        -normalized_distance
    )


def evaluate_rl(
    model: CandidateActor, instances: list[VRPTWInstance]
) -> tuple[Fitness, list[Solution]]:
    model.eval()
    with torch.no_grad():
        solutions = [
            solve(
                instance,
                "rl",
                candidate_selector=_ActorSelector(model, stochastic=False),
            )
            for instance in instances
        ]
    return fitness_from_solutions(instances, solutions), solutions


def train_rl(
    train_instances: list[VRPTWInstance],
    validation_instances: list[VRPTWInstance],
    *,
    seed: int = 0,
    cycles: int = 24,
    episodes_per_cycle: int = 8,
    device: str = "cpu",
    config: RLConfig | None = None,
) -> RLRun:
    """Train a candidate-action policy using episodic REINFORCE."""
    if not train_instances or not validation_instances:
        raise ValueError("Non-empty train and validation sets are required")
    if cycles < 1 or episodes_per_cycle < 2:
        raise ValueError("cycles and at least two episodes per cycle are required")
    config = config or RLConfig()
    seed_everything(seed)
    rng = random.Random(seed)
    torch_device = torch.device(device)
    model = CandidateActor(config.hidden_size).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_fitness: Fitness | None = None
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float]] = []
    validation_interval = max(1, cycles // 4)

    for cycle in range(1, cycles + 1):
        model.train()
        episode_log_probs: list[Tensor] = []
        episode_entropies: list[Tensor] = []
        rewards: list[float] = []
        for _ in range(episodes_per_cycle):
            instance = rng.choice(train_instances)
            selector = _ActorSelector(model, stochastic=True)
            solution = solve(instance, "rl", seed=seed + cycle, candidate_selector=selector)
            if not selector.log_probabilities:
                raise RuntimeError("RL rollout did not record any policy actions")
            episode_log_probs.append(torch.stack(selector.log_probabilities).sum())
            episode_entropies.append(torch.stack(selector.entropies).mean())
            rewards.append(_episode_reward(instance, solution))

        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=torch_device)
        advantages = reward_tensor - reward_tensor.mean()
        reward_scale = reward_tensor.std(unbiased=False)
        if float(reward_scale.detach()) > 1e-8:
            advantages = advantages / reward_scale
        advantages = advantages.detach()
        log_prob_tensor = torch.stack(episode_log_probs)
        entropy = torch.stack(episode_entropies).mean()
        policy_loss = -(advantages * log_prob_tensor).mean()
        loss = policy_loss - config.entropy_coefficient * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        record = {
            "cycle": float(cycle),
            "policy_loss": float(policy_loss.detach().cpu()),
            "mean_reward": float(reward_tensor.mean().cpu()),
            "entropy": float(entropy.detach().cpu()),
        }
        if cycle % validation_interval == 0 or cycle == cycles:
            validation_fitness, _ = evaluate_rl(model, validation_instances)
            record["validation_infeasible"] = float(validation_fitness.infeasible_instances)
            record["validation_vehicles"] = float(validation_fitness.mean_vehicles)
            record["validation_distance"] = float(validation_fitness.mean_normalized_distance)
            if best_fitness is None or validation_fitness < best_fitness:
                best_fitness = validation_fitness
                best_state = copy.deepcopy(model.state_dict())
        history.append(record)

    if best_fitness is None or best_state is None:
        raise RuntimeError("RL training did not produce a validated policy")
    model.load_state_dict(best_state)
    return RLRun(model, best_fitness, history)

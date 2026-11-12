"""Episode fitness shared by GP and GPRT."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from typing import NamedTuple

from gprt_vrptw.evaluator import EvaluationStats
from gprt_vrptw.grammar import DRONE_FEATURES, tree_size, tree_to_string

from .simulator import SIMULATOR_VERSION, run_episode


class DynamicFitness(NamedTuple):
    infeasible_instances: int
    mean_unserved_fraction: float
    mean_normalized_distance: float
    mean_vehicles: float
    complexity: float


class DynamicEvaluator:
    def __init__(self, settings, dispatch, complexity_penalty):
        self.settings = settings
        self.dispatch = dispatch
        self.complexity_penalty = complexity_penalty
        self.stats = EvaluationStats()
        self.cache = {}
        self.runtime_info = {
            "selected": SIMULATOR_VERSION,
            "features": list(DRONE_FEATURES),
            "settings": asdict(settings),
            "dispatch": asdict(dispatch),
        }
        self.config_hash = hashlib.sha256(
            json.dumps(self.runtime_info, sort_keys=True).encode()
        ).hexdigest()

    def solution(self, tree, instance, *, phase="train"):
        requests = f"{phase}_instance_requests"
        evaluations = f"{phase}_instance_evaluations"
        setattr(self.stats, requests, getattr(self.stats, requests) + 1)
        key = tree_to_string(tree), instance.sha256, instance.max_vehicles, self.config_hash
        if key in self.cache:
            self.stats.cache_hits += 1
            return self.cache[key]
        started = time.perf_counter()
        episode = run_episode(instance, self.settings, self.dispatch, tree)
        self.stats.evaluator_seconds += time.perf_counter() - started
        self.cache[key] = episode
        self.stats.unique_evaluations += 1
        self.stats.instance_evaluations += 1
        setattr(self.stats, evaluations, getattr(self.stats, evaluations) + 1)
        return episode

    def fitness(self, tree, instances, *, phase="train"):
        if not instances:
            raise ValueError("Fitness requires at least one instance")
        self.stats.fitness_requests += 1
        episodes = [self.solution(tree, instance, phase=phase) for instance in instances]
        count = len(episodes)
        return DynamicFitness(
            0,  # Hard constraints are enforced before dispatch; integration errors abort.
            sum(1 - result.delivered / item.customer_count
                for result, item in zip(episodes, instances)) / count,
            sum((result.truck_km + self.dispatch.drone_cost_ratio * result.drone_km)
                / item.customer_count for result, item in zip(episodes, instances)) / count,
            sum(result.vehicles for result in episodes) / count,
            tree_size(tree) * self.complexity_penalty,
        )

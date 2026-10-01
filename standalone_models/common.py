from __future__ import annotations

import random

import torch

from gprt_vrptw.evaluator import Fitness, distance_scale
from vrptw_baselines.data import VRPTWInstance
from vrptw_baselines.solver import Solution


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fitness_from_solutions(
    instances: list[VRPTWInstance], solutions: list[Solution]
) -> Fitness:
    if not instances:
        raise ValueError("Cannot score an empty instance set")
    count = len(instances)
    return Fitness(
        sum(not solution.feasible for solution in solutions),
        sum(
            len(solution.unserved) / max(1, instance.customer_count)
            for instance, solution in zip(instances, solutions)
        ) / count,
        sum(solution.vehicle_count for solution in solutions) / count,
        sum(
            solution.distance / (max(1, instance.customer_count) * distance_scale(instance))
            for instance, solution in zip(instances, solutions)
        ) / count,
        0.0,
    )

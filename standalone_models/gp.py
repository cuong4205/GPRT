from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TypeAlias

from vrptw_baselines.data import VRPTWInstance
from vrptw_baselines.solver import CandidateContext, solve


Tree: TypeAlias = str | float | tuple

ARITY = {"add": 2, "sub": 2, "mul": 2, "pdiv": 2, "min": 2, "max": 2}
TERMINALS = (
    "d_current",
    "d_depot",
    "demand",
    "ready",
    "due",
    "slack",
    "wait",
    "remaining_capacity",
    "remaining_fraction",
    "route_load",
    "current_time",
)
CONSTANTS = (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0)


@dataclass(frozen=True, slots=True)
class GPConfig:
    population_size: int = 10
    generations: int = 4
    tournament_size: int = 4
    elite_count: int = 2
    max_depth: int = 5
    crossover_rate: float = 0.65
    mutation_rate: float = 0.25
    validation_candidates: int = 6
    complexity_penalty: float = 1e-4


@dataclass(frozen=True, slots=True)
class GPResult:
    expression: Tree
    expression_text: str
    train_fitness: float
    validation_fitness: float
    fitness_requests: int
    unique_expressions: int
    instance_evaluations: int
    cache_hits: int
    generation_best: tuple[float, ...]


def tree_to_string(tree: Tree) -> str:
    if isinstance(tree, str):
        return tree
    if isinstance(tree, (int, float)):
        return f"{float(tree):g}"
    op = tree[0]
    return f"{op}({', '.join(tree_to_string(child) for child in tree[1:])})"


def tree_size(tree: Tree) -> int:
    if not isinstance(tree, tuple):
        return 1
    return 1 + sum(tree_size(child) for child in tree[1:])


def tree_depth(tree: Tree) -> int:
    if not isinstance(tree, tuple):
        return 0
    return 1 + max(tree_depth(child) for child in tree[1:])


def random_tree(rng: random.Random, max_depth: int, grow: bool = True) -> Tree:
    if max_depth <= 0 or (grow and rng.random() < 0.3):
        return rng.choice(TERMINALS + CONSTANTS)
    op = rng.choice(tuple(ARITY))
    return (op,) + tuple(
        random_tree(rng, max_depth - 1, grow) for _ in range(ARITY[op])
    )


def _paths(tree: Tree, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    paths = [prefix]
    if isinstance(tree, tuple):
        for index, child in enumerate(tree[1:], start=1):
            paths.extend(_paths(child, prefix + (index,)))
    return paths


def _at(tree: Tree, path: tuple[int, ...]) -> Tree:
    node = tree
    for index in path:
        assert isinstance(node, tuple)
        node = node[index]
    return node


def _replace(tree: Tree, path: tuple[int, ...], replacement: Tree) -> Tree:
    if not path:
        return replacement
    assert isinstance(tree, tuple)
    index = path[0]
    values = list(tree)
    values[index] = _replace(values[index], path[1:], replacement)
    return tuple(values)


def crossover(a: Tree, b: Tree, rng: random.Random, max_depth: int) -> Tree:
    child = _replace(a, rng.choice(_paths(a)), _at(b, rng.choice(_paths(b))))
    return child if tree_depth(child) <= max_depth else a


def mutate(tree: Tree, rng: random.Random, max_depth: int) -> Tree:
    path = rng.choice(_paths(tree))
    remaining_depth = max(0, max_depth - len(path))
    child = _replace(tree, path, random_tree(rng, remaining_depth, grow=True))
    return child if tree_depth(child) <= max_depth else tree


_DISTANCE_SCALE_CACHE: dict[str, float] = {}


def _distance_scale(instance: VRPTWInstance) -> float:
    cached = _DISTANCE_SCALE_CACHE.get(instance.sha256)
    if cached is None:
        cached = max(
            1.0,
            max(instance.distance(0, customer_id) for customer_id in range(1, len(instance.customers))),
        )
        _DISTANCE_SCALE_CACHE[instance.sha256] = cached
    return cached


def features(instance: VRPTWInstance, context: CandidateContext) -> dict[str, float]:
    customer = instance.customers[context.customer_id]
    depot_due = max(1.0, instance.depot.due_date)
    scale = _distance_scale(instance)
    distance_current = instance.distance(context.current_id, context.customer_id)
    return {
        "d_current": distance_current / scale,
        "d_depot": instance.distance(context.customer_id, 0) / scale,
        "demand": customer.demand / max(1.0, instance.capacity),
        "ready": customer.ready_time / depot_due,
        "due": customer.due_date / depot_due,
        "slack": max(0.0, customer.due_date - context.arrival_time) / depot_due,
        "wait": max(0.0, customer.ready_time - context.arrival_time) / depot_due,
        "remaining_capacity": context.remaining_capacity / max(1.0, instance.capacity),
        "remaining_fraction": context.unserved_count / max(1, instance.customer_count),
        "route_load": (instance.capacity - context.remaining_capacity) / max(1.0, instance.capacity),
        "current_time": context.current_time / depot_due,
    }


def evaluate_tree(tree: Tree, values: dict[str, float]) -> float:
    if isinstance(tree, str):
        return values[tree]
    if isinstance(tree, (int, float)):
        return float(tree)
    op = tree[0]
    left = evaluate_tree(tree[1], values)
    right = evaluate_tree(tree[2], values)
    if op == "add":
        value = left + right
    elif op == "sub":
        value = left - right
    elif op == "mul":
        value = left * right
    elif op == "pdiv":
        value = left if abs(right) < 1e-9 else left / right
    elif op == "min":
        value = min(left, right)
    elif op == "max":
        value = max(left, right)
    else:
        raise ValueError(f"Unknown GP operator: {op}")
    if not math.isfinite(value):
        return 1e6
    return max(-1e6, min(1e6, value))


def make_priority(tree: Tree):
    return lambda instance, context: evaluate_tree(tree, features(instance, context))


def _fitness(
    tree: Tree,
    instances: list[VRPTWInstance],
    penalty: float,
    evaluation_cache: dict[tuple[str, str], float] | None = None,
) -> tuple[float, int, int]:
    total = 0.0
    priority = make_priority(tree)
    expression = tree_to_string(tree)
    evaluations = 0
    cache_hits = 0
    for instance in instances:
        key = (expression, instance.sha256)
        if evaluation_cache is not None and key in evaluation_cache:
            objective = evaluation_cache[key]
            cache_hits += 1
        else:
            objective = solve(instance, "gp", priority_function=priority).objective
            evaluations += 1
            if evaluation_cache is not None:
                evaluation_cache[key] = objective
        total += objective
    score = total / max(1, len(instances)) + penalty * tree_size(tree)
    return score, evaluations, cache_hits


def evolve(
    train_instances: list[VRPTWInstance],
    validation_instances: list[VRPTWInstance],
    *,
    seed: int,
    config: GPConfig = GPConfig(),
) -> GPResult:
    if not train_instances or not validation_instances:
        raise ValueError("GP needs non-empty train and validation sets")
    rng = random.Random(seed)
    population: list[Tree] = list(TERMINALS[: config.population_size])
    population.extend(
        random_tree(rng, config.max_depth, grow=index % 2 == 0)
        for index in range(config.population_size - len(population))
    )
    cache: dict[str, float] = {}
    evaluation_cache: dict[tuple[str, str], float] = {}
    fitness_requests = 0
    instance_evaluations = 0
    cache_hits = 0
    generation_best: list[float] = []

    def fitness(tree: Tree) -> float:
        nonlocal fitness_requests, instance_evaluations, cache_hits
        fitness_requests += 1
        key = tree_to_string(tree)
        if key not in cache:
            score, evaluated, hits = _fitness(
                tree, train_instances, config.complexity_penalty, evaluation_cache
            )
            cache[key] = score
            instance_evaluations += evaluated
            cache_hits += hits
        return cache[key]

    archive: dict[str, tuple[Tree, float]] = {}
    for _ in range(config.generations):
        scored = sorted(((fitness(tree), tree) for tree in population), key=lambda item: item[0])
        generation_best.append(scored[0][0])
        for score, tree in scored:
            archive[tree_to_string(tree)] = (tree, score)

        def tournament() -> Tree:
            contestants = rng.sample(scored, min(config.tournament_size, len(scored)))
            return min(contestants, key=lambda item: item[0])[1]

        next_population = [tree for _, tree in scored[: config.elite_count]]
        while len(next_population) < config.population_size:
            draw = rng.random()
            parent = tournament()
            if draw < config.crossover_rate:
                child = crossover(parent, tournament(), rng, config.max_depth)
            elif draw < config.crossover_rate + config.mutation_rate:
                child = mutate(parent, rng, config.max_depth)
            else:
                child = parent
            next_population.append(child)
        population = next_population

    finalists = sorted(archive.values(), key=lambda item: item[1])[
        : config.validation_candidates
    ]
    validated = []
    for tree, train_score in finalists:
        score, evaluated, hits = _fitness(
            tree, validation_instances, config.complexity_penalty, evaluation_cache
        )
        instance_evaluations += evaluated
        cache_hits += hits
        validated.append((score, train_score, tree))
    validation_score, train_score, best = min(validated, key=lambda item: item[0])
    return GPResult(
        expression=best,
        expression_text=tree_to_string(best),
        train_fitness=train_score,
        validation_fitness=validation_score,
        fitness_requests=fitness_requests,
        unique_expressions=len(cache),
        instance_evaluations=instance_evaluations,
        cache_hits=cache_hits,
        generation_best=tuple(generation_best),
    )

from __future__ import annotations

import random

from vrptw_baselines.data import VRPTWInstance

from .config import GPRTConfig
from .evaluator import Fitness, FitnessEvaluator
from .grammar import Grammar, Tree, prefix_to_tree, simplify_tree, tree_to_prefix, tree_to_string


def _paths(tree: Tree, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    paths = [prefix]
    if isinstance(tree, tuple):
        for index, child in enumerate(tree[1:], 1):
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
    values = list(tree)
    values[path[0]] = _replace(values[path[0]], path[1:], replacement)
    return tuple(values)


def valid(tree: Tree, grammar: Grammar) -> bool:
    return grammar.validate_expression(tree_to_prefix(tree))


def crossover(a: Tree, b: Tree, rng: random.Random, grammar: Grammar) -> Tree:
    for _ in range(20):
        path = rng.choice(_paths(a))
        donors = _paths(b)
        child = simplify_tree(_replace(a, path, _at(b, rng.choice(donors))))
        if valid(child, grammar):
            return child
    return a
def mutate(tree: Tree, rng: random.Random, grammar: Grammar) -> Tree:
    for _ in range(20):
        path = rng.choice(_paths(tree))
        replacement = grammar.random_tree(rng, grow=True)
        child = simplify_tree(_replace(tree, path, replacement))
        if valid(child, grammar):
            return child
    return tree
def deduplicate_fill(population: list[Tree], size: int, rng: random.Random, grammar: Grammar) -> list[Tree]:
    unique: dict[str, Tree] = {}
    for tree in population:
        tree = simplify_tree(tree)
        if not valid(tree, grammar):
            raise ValueError("GP seed violates grammar after simplification")
        unique.setdefault(tree_to_string(tree), tree)
    attempts = 0
    while len(unique) < size and attempts < size * 100:
        tree = simplify_tree(grammar.random_tree(rng, grow=bool(attempts % 2)))
        unique.setdefault(tree_to_string(tree), tree)
        attempts += 1
    values = list(unique.values())
    while len(values) < size:  # tiny grammars/tests can legitimately exhaust uniqueness
        values.append(grammar.random_tree(rng, grow=True))
    return values[:size]


def evolve_population(
    seeds: list[Tree],
    instances: list[VRPTWInstance],
    evaluator: FitnessEvaluator,
    rng: random.Random,
    grammar: Grammar,
    config: GPRTConfig,
) -> tuple[list[Tree], dict[str, Fitness], list[Fitness]]:
    population = deduplicate_fill(seeds, config.population_size, rng, grammar)
    archive: dict[str, Fitness] = {}
    generation_best: list[Fitness] = []

    for _ in range(config.gp_generations):
        scored_pairs = [(evaluator.fitness(tree, instances), tree) for tree in population]
        scored_pairs.sort(key=lambda item: item[0])
        generation_best.append(scored_pairs[0][0])
        for fitness, tree in scored_pairs:
            key = tree_to_string(tree)
            if key not in archive or fitness < archive[key]:
                archive[key] = fitness

        def tournament() -> Tree:
            contestants = rng.sample(scored_pairs, min(config.tournament_size, len(scored_pairs)))
            return min(contestants, key=lambda item: item[0])[1]

        next_population = [tree for _, tree in scored_pairs[: config.elite_count]]
        while len(next_population) < config.population_size:
            draw = rng.random()
            parent = tournament()
            if draw < config.crossover_rate:
                child = crossover(parent, tournament(), rng, grammar)
            elif draw < config.crossover_rate + config.mutation_rate:
                child = mutate(parent, rng, grammar)
            else:
                child = parent
            next_population.append(child)
        population = deduplicate_fill(next_population, config.population_size, rng, grammar)

    final_pairs = [(evaluator.fitness(tree, instances), tree) for tree in population]
    final_pairs.sort(key=lambda item: item[0])
    for fitness, tree in final_pairs:
        archive[tree_to_string(tree)] = min(archive.get(tree_to_string(tree), fitness), fitness)
    return population, archive, generation_best

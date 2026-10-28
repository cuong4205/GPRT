from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from gprt_vrptw import fast_routing
from gprt_vrptw.evaluator import FitnessEvaluator, compile_tree, make_priority
from gprt_vrptw.grammar import ARITY, FEATURES, Grammar
from gprt_vrptw.splits import split_solomon, stratified_sample
from gprt_vrptw.trainer import GPRTTrainer
from vrptw_baselines.data import Customer, VRPTWInstance, discover_instances
from vrptw_baselines.solver import solve
from test_gprt_vrptw import tiny_config, tiny_instance


def test_python_fallback_and_explicit_backend_selection():
    with patch.object(fast_routing, "njit", None):
        priority = make_priority("d_current", backend="auto")
        assert priority.route_backend == "python"
        assert not hasattr(priority, "construct_routes")
        with pytest.raises(RuntimeError, match="unavailable"):
            make_priority("d_current", backend="numba")
    with pytest.raises(ValueError, match="backend"):
        make_priority("d_current", backend="invalid")


native = pytest.mark.skipif(not fast_routing.available(), reason="optional Numba backend not installed")


@native
def test_native_interpreter_matches_all_operators_and_protected_arithmetic():
    rng = random.Random(2810)
    grammar = Grammar(63, 6)
    trees = [*FEATURES, *(grammar.random_tree(rng) for _ in range(150)),
             *((op, 0.0, 1.0, -1.0) if arity == 3 else (op, "d_current", "d_depot")
               for op, arity in ARITY.items()),
             ("pdiv", 1.0, ("sub", "d_current", "d_depot")),
             ("mul", ("mul", "d_current", "d_depot"), "d_current")]
    vectors = [np.asarray([rng.uniform(-2, 2) for _ in FEATURES]) for _ in range(20)]
    vectors += [np.full(len(FEATURES), value, dtype=np.float64)
                for value in (0, -0.0, 1e-10, 1e300, -1e300, float("nan"), float("inf"), -float("inf"))]
    for tree in trees:
        codes, constants, _ = fast_routing.pack_program(tree)
        reference = compile_tree(tree)
        for vector in vectors:
            actual = fast_routing._program_score(codes, constants, vector, np.empty(len(codes)))
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                expected = reference(vector)
            assert actual == expected, (tree, vector, actual, expected)


@native
def test_native_routes_and_all_solution_metrics_match_on_train_families():
    instances = discover_instances("data/data/Solomon")
    split = split_solomon(instances)
    train = stratified_sample([i for i in instances if split[i.name.lower()] == "train"], 6)
    grammar, rng = Grammar(31, 4), random.Random(28)
    trees = [*FEATURES, *(grammar.random_tree(rng) for _ in range(8)),
             ("if_else", ("le", "slack", "wait_time"), "urgency_rank", "next_feasible_fraction")]
    for tree in trees:
        python, compiled = (make_priority(tree, backend=mode) for mode in ("python", "numba"))
        for instance in train:
            assert solve(instance, "gp", priority_function=compiled) == solve(instance, "gp", priority_function=python)


@native
def test_native_boundary_infeasible_capacity_depot_and_exact_ties():
    original = tiny_instance()
    cases = [original, replace(original, max_vehicles=1), replace(original, capacity=.1),
             replace(original, capacity=1),
             replace(original, customers=(replace(original.depot, ready_time=10), *original.customers[1:])),
             replace(original, customers=(original.depot,
                 Customer(1, 1, 0, 1, 0, 1, 1),
                 Customer(2, 2, 0, 1, 0, 2, 1),
                 Customer(3, 3, 0, 1, 0, 3 - 2e-9, 1)))]
    for instance in cases:
        for tree in (0.0, -1.0, "remaining_vehicles", "next_feasible_fraction", "urgency_rank"):
            assert solve(instance, "gp", priority_function=make_priority(tree, backend="numba")) == solve(
                instance, "gp", priority_function=make_priority(tree, backend="python"))


@native
def test_acceleration_still_uses_independent_checker_and_does_not_change_budget():
    instance = tiny_instance()
    from vrptw_baselines.solver import validate_solution

    with patch("vrptw_baselines.solver.validate_solution", wraps=validate_solution) as checker:
        result = solve(instance, "gp", priority_function=make_priority("d_current", backend="numba"))
        assert checker.call_count == 1 and result.feasible
    with patch.dict("os.environ", {"GPRT_ROUTE_BACKEND": "numba"}):
        fast = FitnessEvaluator()
        fast_fitness = fast.fitness("d_current", [instance])
        fast.fitness("d_current", [instance])
    with patch.dict("os.environ", {"GPRT_ROUTE_BACKEND": "python"}):
        slow = FitnessEvaluator()
        slow_fitness = slow.fitness("d_current", [instance])
        slow.fitness("d_current", [instance])
    assert fast_fitness == slow_fitness
    assert fast.stats.fitness_requests == slow.stats.fitness_requests == 2
    assert fast.stats.unique_evaluations == slow.stats.unique_evaluations == 1
    assert fast.stats.cache_hits == slow.stats.cache_hits == 1


@native
def test_fractional_randomized_instances_match_and_matrix_fallback_is_safe():
    rng = random.Random(14028)
    grammar = Grammar(31, 4)
    trees = [*FEATURES, *(grammar.random_tree(rng) for _ in range(6))]
    for case in range(20):
        count = rng.randint(2, 12)
        customers = [Customer(0, 0, 0, 0, rng.choice((0, 5)), 70, 0)]
        for cid in range(1, count + 1):
            ready = rng.uniform(0, 30)
            customers.append(Customer(cid, rng.uniform(-10, 10), rng.uniform(-10, 10),
                                      rng.uniform(.01, 1.5), ready, ready + rng.uniform(0, 30), rng.uniform(0, 5)))
        instance = VRPTWInstance(f"fractional{case}", Path("synthetic.txt"), rng.randint(1, 5),
                                 rng.uniform(.1, 4), tuple(customers), f"fractional{case}", "synthetic")
        for tree in trees:
            assert solve(instance, "gp", priority_function=make_priority(tree, backend="numba")) == solve(
                instance, "gp", priority_function=make_priority(tree, backend="python"))
    instance = tiny_instance()
    with patch.object(fast_routing, "_inputs", return_value=None):
        assert solve(instance, "gp", priority_function=make_priority("d_current", backend="numba")) == solve(
            instance, "gp", priority_function=make_priority("d_current", backend="python"))


@native
def test_gprt_learning_trajectory_and_model_tensors_match_across_backends(tmp_path):
    trainers, results = [], []
    for backend in ("python", "numba"):
        with patch.dict("os.environ", {"GPRT_ROUTE_BACKEND": backend}):
            trainer = GPRTTrainer(tiny_config(cycles=2))
            result = trainer.train([tiny_instance("train", "train")], [tiny_instance("val", "val")],
                                   seed=28, output_dir=tmp_path / backend,
                                   split_manifest={"train": "train", "val": "validation"})
            trainers.append(trainer)
            results.append(result)
            checkpoint = torch.load(tmp_path / backend / "last.pt", weights_only=False)
            assert checkpoint["routing_runtime"]["selected"] == backend
    assert results[0].best_tree == results[1].best_tree
    assert results[0].best_validation_fitness == results[1].best_validation_fitness
    for key, tensor in trainers[0].model.state_dict().items():
        assert torch.equal(tensor, trainers[1].model.state_dict()[key]), key
    for left, right in zip(results[0].logs, results[1].logs):
        for key in ("loss", "policy_loss", "elite_nll", "train_fitness", "fitness_requests", "unique_evaluations"):
            assert left[key] == right[key], key

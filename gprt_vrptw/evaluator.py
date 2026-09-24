from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, NamedTuple

from vrptw_baselines.data import VRPTWInstance
from vrptw_baselines.solver import CandidateContext, Solution, solve

from .grammar import FEATURES, Tree, tree_size, tree_to_string


class Fitness(NamedTuple):
    infeasible_instances: int
    mean_unserved_fraction: float
    mean_vehicles: float
    mean_normalized_distance: float
    complexity: float


@dataclass(slots=True)
class EvaluationStats:
    fitness_requests: int = 0
    unique_evaluations: int = 0
    cache_hits: int = 0
    instance_evaluations: int = 0
    train_instance_requests: int = 0
    screening_instance_requests: int = 0
    validation_instance_requests: int = 0
    diagnostic_instance_requests: int = 0
    train_instance_evaluations: int = 0
    screening_instance_evaluations: int = 0
    validation_instance_evaluations: int = 0
    diagnostic_instance_evaluations: int = 0
    evaluator_seconds: float = 0.0


_SCALE_CACHE: dict[str, float] = {}


def distance_scale(instance: VRPTWInstance) -> float:
    if instance.sha256 not in _SCALE_CACHE:
        _SCALE_CACHE[instance.sha256] = max(
            1.0,
            max(instance.distance(0, i) for i in range(1, len(instance.customers))),
        )
    return _SCALE_CACHE[instance.sha256]


def normalized_features(
    instance: VRPTWInstance,
    context: CandidateContext,
    *,
    use_lookahead_features: bool = True,
) -> dict[str, float]:
    customer = instance.customers[context.customer_id]
    horizon = max(1.0, instance.depot.due_date)
    capacity = max(1.0, instance.capacity)
    scale = distance_scale(instance)
    route_load = instance.capacity - context.remaining_capacity
    next_feasible = 0
    candidate = instance.customers[context.customer_id]
    departure = context.service_start + candidate.service_time
    capacity_after = context.remaining_capacity - candidate.demand
    if use_lookahead_features:
        for other_id in context.unserved_ids:
            if other_id == context.customer_id:
                continue
            other = instance.customers[other_id]
            if other.demand > capacity_after + 1e-9:
                continue
            arrival = departure + instance.distance(context.customer_id, other_id)
            service_start = max(arrival, other.ready_time)
            if service_start > other.due_date + 1e-9:
                continue
            if service_start + other.service_time + instance.distance(other_id, 0) > instance.depot.due_date + 1e-9:
                continue
            next_feasible += 1
    urgency_rank = (sum(
        instance.customers[other_id].due_date <= customer.due_date
        for other_id in context.unserved_ids
    ) / max(1, context.unserved_count)) if use_lookahead_features else 0.0
    return {
        "d_current": instance.distance(context.current_id, context.customer_id) / scale,
        "d_depot": instance.distance(context.customer_id, 0) / scale,
        "demand": customer.demand / capacity,
        "ready_time": customer.ready_time / horizon,
        "due_date": customer.due_date / horizon,
        "service_time": customer.service_time / horizon,
        "arrival_time": context.arrival_time / horizon,
        "slack": (customer.due_date - context.arrival_time) / horizon,
        "wait_time": max(0.0, customer.ready_time - context.arrival_time) / horizon,
        "remaining_capacity": context.remaining_capacity / capacity,
        "route_load": route_load / capacity,
        "remaining_fraction": context.unserved_count / max(1, instance.customer_count),
        "current_time": context.current_time / horizon,
        "remaining_vehicles": (
            max(0, instance.max_vehicles - context.routes_used) / max(1, instance.max_vehicles)
            if use_lookahead_features else 0.0
        ),
        "next_feasible_fraction": (
            next_feasible / max(1, context.unserved_count - 1)
            if use_lookahead_features else 0.0
        ),
        "urgency_rank": urgency_rank if use_lookahead_features else 0.0,
    }


def normalized_feature_vector(
    instance: VRPTWInstance,
    context: CandidateContext,
    *,
    use_lookahead_features: bool = True,
) -> tuple[float, ...]:
    values = normalized_features(instance, context, use_lookahead_features=use_lookahead_features)
    return tuple(values[name] for name in FEATURES)


def _bounded(value: float, limit: float = 1e6) -> float:
    if not math.isfinite(value):
        return math.copysign(limit, value if not math.isnan(value) else 1.0)
    return max(-limit, min(limit, value))


def _add(a: float, b: float) -> float: return _bounded(a + b)
def _sub(a: float, b: float) -> float: return _bounded(a - b)
def _mul(a: float, b: float) -> float: return _bounded(a * b)
def _pdiv(a: float, b: float) -> float: return _bounded(a if abs(b) < 1e-9 else a / b)
def _min(a: float, b: float) -> float: return min(a, b)
def _max(a: float, b: float) -> float: return max(a, b)
def _ge(a: float, b: float) -> float: return float(a >= b)
def _le(a: float, b: float) -> float: return float(a <= b)
def _and(a: float, b: float) -> float: return float(bool(a) and bool(b))
def _or(a: float, b: float) -> float: return float(bool(a) or bool(b))
def _if_else(condition: float, true_value: float, false_value: float) -> float:
    return true_value if bool(condition) else false_value


_COMPILE_GLOBALS = {
    "_add": _add, "_sub": _sub, "_mul": _mul, "_pdiv": _pdiv,
    "_min": _min, "_max": _max, "_ge": _ge, "_le": _le,
    "_and": _and, "_or": _or, "_if_else": _if_else,
}
_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURES)}
_COMPILED_CACHE: dict[str, Any] = {}


def _tree_source(tree: Tree) -> str:
    if isinstance(tree, str):
        return f"f[{_FEATURE_INDEX[tree]}]"
    if isinstance(tree, (int, float)):
        return repr(float(tree))
    op = str(tree[0])
    return f"_{op}({', '.join(_tree_source(child) for child in tree[1:])})"


def compile_tree(tree: Tree):
    """Compile a grammar-validated tree to fast, still-protected Python bytecode."""
    expression = tree_to_string(tree)
    if expression not in _COMPILED_CACHE:
        _COMPILED_CACHE[expression] = eval(
            f"lambda f: _bounded({_tree_source(tree)})",
            {"__builtins__": {}, "_bounded": _bounded, **_COMPILE_GLOBALS},
        )
    return _COMPILED_CACHE[expression]


class _LazyFeatureVector:
    """Compute and memoize only terminals accessed by the compiled expression.

    The dense normalized_features API remains the reference/debug path. Values
    are local to one candidate and are never reused across routing states.
    """

    __slots__ = ("instance", "context", "lookahead", "values")

    def __init__(self, instance: VRPTWInstance, context: CandidateContext, lookahead: bool):
        self.instance = instance
        self.context = context
        self.lookahead = lookahead
        self.values: dict[int, float] = {}

    def __getitem__(self, index: int) -> float:
        if index not in self.values:
            self.values[index] = self._compute(FEATURES[index])
        return self.values[index]

    def _compute(self, name: str) -> float:
        instance, context = self.instance, self.context
        customer = instance.customers[context.customer_id]
        if name == "d_current":
            return instance.distance(context.current_id, context.customer_id) / distance_scale(instance)
        if name == "d_depot":
            return instance.distance(context.customer_id, 0) / distance_scale(instance)
        if name == "demand":
            return customer.demand / max(1.0, instance.capacity)
        if name == "remaining_capacity":
            return context.remaining_capacity / max(1.0, instance.capacity)
        if name == "route_load":
            return (instance.capacity - context.remaining_capacity) / max(1.0, instance.capacity)
        if name == "remaining_fraction":
            return context.unserved_count / max(1, instance.customer_count)
        if name in {"ready_time", "due_date", "service_time"}:
            return getattr(customer, name) / max(1.0, instance.depot.due_date)
        if name in {"arrival_time", "current_time"}:
            return getattr(context, name) / max(1.0, instance.depot.due_date)
        if name == "slack":
            return (customer.due_date - context.arrival_time) / max(1.0, instance.depot.due_date)
        if name == "wait_time":
            return max(0.0, customer.ready_time - context.arrival_time) / max(1.0, instance.depot.due_date)
        if not self.lookahead:
            return 0.0
        if name == "remaining_vehicles":
            return max(0, instance.max_vehicles - context.routes_used) / max(1, instance.max_vehicles)
        if name == "urgency_rank":
            return sum(
                instance.customers[other_id].due_date <= customer.due_date
                for other_id in context.unserved_ids
            ) / max(1, context.unserved_count)
        if name == "next_feasible_fraction":
            departure = context.service_start + customer.service_time
            capacity_after = context.remaining_capacity - customer.demand
            next_feasible = 0
            for other_id in context.unserved_ids:
                if other_id == context.customer_id:
                    continue
                other = instance.customers[other_id]
                if other.demand > capacity_after + 1e-9:
                    continue
                arrival = departure + instance.distance(context.customer_id, other_id)
                service_start = max(arrival, other.ready_time)
                if service_start > other.due_date + 1e-9:
                    continue
                if service_start + other.service_time + instance.distance(other_id, 0) > instance.depot.due_date + 1e-9:
                    continue
                next_feasible += 1
            return next_feasible / max(1, context.unserved_count - 1)
        raise KeyError(name)


def make_priority(tree: Tree, *, use_lookahead_features: bool = True):
    compiled = compile_tree(tree)

    def priority(instance: VRPTWInstance, context: CandidateContext) -> float:
        return compiled(
            _LazyFeatureVector(instance, context, use_lookahead_features)
        )

    return priority


class FitnessEvaluator:
    def __init__(self, complexity_penalty: float = 1e-4, *, use_lookahead_features: bool = True):
        self.complexity_penalty = complexity_penalty
        self.use_lookahead_features = use_lookahead_features
        self.stats = EvaluationStats()
        self.cache: dict[tuple[str, str, str], Solution] = {}
        self.config_hash = hashlib.sha256(
            json.dumps(
                {
                    "distance": "euclidean_unrounded",
                    "tie_break": "customer_id",
                    "feature_version": 2 if use_lookahead_features else 1,
                    "solver": "shared_constructive_v2",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def solution(
        self,
        tree: Tree,
        instance: VRPTWInstance,
        *,
        phase: str = "train",
    ) -> Solution:
        request_counter = f"{phase}_instance_requests"
        evaluation_counter = f"{phase}_instance_evaluations"
        if not hasattr(self.stats, request_counter):
            raise ValueError(f"Unknown evaluator phase: {phase}")
        setattr(
            self.stats,
            request_counter,
            getattr(self.stats, request_counter) + 1,
        )
        expression = tree_to_string(tree)
        key = (expression, instance.sha256, self.config_hash)
        if key in self.cache:
            self.stats.cache_hits += 1
            return self.cache[key]
        started = time.perf_counter()
        solution = solve(
            instance,
            "gp",
            priority_function=make_priority(
                tree, use_lookahead_features=self.use_lookahead_features
            ),
        )
        self.stats.evaluator_seconds += time.perf_counter() - started
        self.cache[key] = solution
        self.stats.unique_evaluations += 1
        self.stats.instance_evaluations += 1
        setattr(
            self.stats,
            evaluation_counter,
            getattr(self.stats, evaluation_counter) + 1,
        )
        return solution

    def fitness(
        self,
        tree: Tree,
        instances: list[VRPTWInstance],
        *,
        phase: str = "train",
    ) -> Fitness:
        if not instances:
            raise ValueError("Fitness requires at least one instance")
        self.stats.fitness_requests += 1
        solutions = [self.solution(tree, instance, phase=phase) for instance in instances]
        count = len(instances)
        return Fitness(
            sum(not s.feasible for s in solutions),
            sum(len(s.unserved) / i.customer_count for s, i in zip(solutions, instances)) / count,
            sum(s.vehicle_count for s in solutions) / count,
            sum(s.distance / (max(1, i.customer_count) * distance_scale(i)) for s, i in zip(solutions, instances)) / count,
            tree_size(tree) * self.complexity_penalty,
        )


def rank_rewards(fitnesses: list[Fitness]) -> list[float]:
    """Rank rewards in [-1, 1], best=1; ties receive their average rank."""
    if not fitnesses:
        return []
    if len(fitnesses) == 1:
        return [0.0]
    order = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
    ranks = [0.0] * len(fitnesses)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and fitnesses[order[end]] == fitnesses[order[pos]]:
            end += 1
        average_rank = (pos + end - 1) / 2
        for index in order[pos:end]:
            ranks[index] = average_rank
        pos = end
    return [1.0 - 2.0 * rank / (len(fitnesses) - 1) for rank in ranks]


def behavior_fingerprint(
    tree: Tree,
    instances: list[VRPTWInstance],
    evaluator: FitnessEvaluator,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Observed route decisions on a fixed probe; not a global equivalence proof."""
    return tuple(
        evaluator.solution(tree, instance, phase="diagnostic").routes
        for instance in instances
    )

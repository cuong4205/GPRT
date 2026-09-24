from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from .data import VRPTWInstance


@dataclass(frozen=True, slots=True)
class CandidateContext:
    customer_id: int
    current_id: int
    current_time: float
    arrival_time: float
    service_start: float
    remaining_capacity: float
    unserved_count: int
    unserved_ids: tuple[int, ...] = ()
    routes_used: int = 1


@dataclass(frozen=True, slots=True)
class RouteMetrics:
    route_id: int
    customers: tuple[int, ...]
    load: float
    distance: float
    end_time: float
    waiting_time: float
    feasible: bool


@dataclass(frozen=True, slots=True)
class Solution:
    method: str
    routes: tuple[tuple[int, ...], ...]
    feasible: bool
    vehicle_count: int
    distance: float
    waiting_time: float
    objective: float
    unserved: tuple[int, ...]
    route_metrics: tuple[RouteMetrics, ...]


PriorityFunction = Callable[[VRPTWInstance, CandidateContext], float]


def _candidate_contexts(
    instance: VRPTWInstance,
    unserved: set[int],
    current_id: int,
    current_time: float,
    remaining_capacity: float,
    routes_used: int,
) -> list[CandidateContext]:
    contexts: list[CandidateContext] = []
    depot = instance.depot
    unserved_ids = tuple(sorted(unserved))
    for customer_id in unserved_ids:
        customer = instance.customers[customer_id]
        if customer.demand > remaining_capacity + 1e-9:
            continue
        arrival = current_time + instance.distance(current_id, customer_id)
        service_start = max(arrival, customer.ready_time)
        if service_start > customer.due_date + 1e-9:
            continue
        departure = service_start + customer.service_time
        if departure + instance.distance(customer_id, 0) > depot.due_date + 1e-9:
            continue
        contexts.append(
            CandidateContext(
                customer_id=customer_id,
                current_id=current_id,
                current_time=current_time,
                arrival_time=arrival,
                service_start=service_start,
                remaining_capacity=remaining_capacity,
                unserved_count=len(unserved),
                unserved_ids=unserved_ids,
                routes_used=routes_used,
            )
        )
    return contexts


def solve(
    instance: VRPTWInstance,
    method: str,
    *,
    seed: int = 0,
    priority_function: PriorityFunction | None = None,
) -> Solution:
    """Construct a solution using one shared feasibility engine.

    Supported methods are fifo, random, greedy, edd, and gp. Lower GP priority
    values are selected first. Ties are resolved by due date then customer ID.
    """

    normalized_method = method.lower().replace("-", "_")
    if normalized_method not in {"fifo", "random", "greedy", "edd", "gp"}:
        raise ValueError(f"Unknown method: {method}")
    if normalized_method == "gp" and priority_function is None:
        raise ValueError("GP requires a priority_function")

    rng = random.Random(seed)
    unserved = set(range(1, len(instance.customers)))
    routes: list[tuple[int, ...]] = []

    while unserved and len(routes) < instance.max_vehicles:
        current_id = 0
        current_time = max(0.0, instance.depot.ready_time)
        remaining_capacity = instance.capacity
        route = [0]

        while True:
            contexts = _candidate_contexts(
                instance,
                unserved,
                current_id,
                current_time,
                remaining_capacity,
                len(routes) + 1,
            )
            if not contexts:
                break

            if normalized_method == "random":
                chosen = rng.choice(contexts)
            elif normalized_method == "fifo":
                chosen = min(contexts, key=lambda context: context.customer_id)
            elif normalized_method == "greedy":
                chosen = min(
                    contexts,
                    key=lambda context: (
                        instance.distance(context.current_id, context.customer_id),
                        context.customer_id,
                    ),
                )
            elif normalized_method == "edd":
                chosen = min(
                    contexts,
                    key=lambda context: (
                        instance.customers[context.customer_id].due_date,
                        context.customer_id,
                    ),
                )
            else:
                assert priority_function is not None
                chosen = min(
                    contexts,
                    key=lambda context: (
                        _finite(priority_function(instance, context)),
                        context.customer_id,
                    ),
                )

            customer = instance.customers[chosen.customer_id]
            route.append(chosen.customer_id)
            unserved.remove(chosen.customer_id)
            remaining_capacity -= customer.demand
            current_time = chosen.service_start + customer.service_time
            current_id = chosen.customer_id

        if len(route) == 1:
            break
        route.append(0)
        routes.append(tuple(route))

    checked = validate_solution(instance, routes)
    feasible = checked["feasible"] and not unserved
    vehicle_count = len(routes)
    distance = float(checked["distance"])
    objective = (0.0 if feasible else 1e12) + vehicle_count * 1e6 + distance
    return Solution(
        method=normalized_method,
        routes=tuple(routes),
        feasible=feasible,
        vehicle_count=vehicle_count,
        distance=distance,
        waiting_time=float(checked["waiting_time"]),
        objective=objective,
        unserved=tuple(sorted(unserved)),
        route_metrics=tuple(checked["route_metrics"]),
    )


def _finite(value: float) -> float:
    if not math.isfinite(value):
        return 1e12
    return max(-1e12, min(1e12, value))


def validate_solution(
    instance: VRPTWInstance, routes: list[tuple[int, ...]] | tuple[tuple[int, ...], ...]
) -> dict[str, object]:
    """Independently recompute distance, timing, capacity, and visit coverage."""

    expected = set(range(1, len(instance.customers)))
    visited: list[int] = []
    total_distance = 0.0
    total_waiting = 0.0
    route_metrics: list[RouteMetrics] = []
    globally_feasible = len(routes) <= instance.max_vehicles

    for route_id, route in enumerate(routes, start=1):
        route_feasible = len(route) >= 3 and route[0] == 0 and route[-1] == 0
        load = 0.0
        distance = 0.0
        waiting = 0.0
        current_time = max(0.0, instance.depot.ready_time)
        current_id = 0

        for customer_id in route[1:]:
            if customer_id < 0 or customer_id >= len(instance.customers):
                route_feasible = False
                continue
            travel = instance.distance(current_id, customer_id)
            distance += travel
            arrival = current_time + travel
            customer = instance.customers[customer_id]
            if customer_id == 0:
                service_start = max(arrival, instance.depot.ready_time)
                if service_start > instance.depot.due_date + 1e-9:
                    route_feasible = False
                current_time = service_start
            else:
                visited.append(customer_id)
                load += customer.demand
                service_start = max(arrival, customer.ready_time)
                waiting += max(0.0, customer.ready_time - arrival)
                if service_start > customer.due_date + 1e-9:
                    route_feasible = False
                current_time = service_start + customer.service_time
            current_id = customer_id

        if load > instance.capacity + 1e-9:
            route_feasible = False
        total_distance += distance
        total_waiting += waiting
        globally_feasible = globally_feasible and route_feasible
        route_metrics.append(
            RouteMetrics(
                route_id=route_id,
                customers=tuple(route),
                load=load,
                distance=distance,
                end_time=current_time,
                waiting_time=waiting,
                feasible=route_feasible,
            )
        )

    coverage_ok = set(visited) == expected and len(visited) == len(expected)
    return {
        "feasible": globally_feasible and coverage_ok,
        "coverage_ok": coverage_ok,
        "distance": total_distance,
        "waiting_time": total_waiting,
        "route_metrics": route_metrics,
        "missing": sorted(expected - set(visited)),
        "duplicates": sorted({customer for customer in visited if visited.count(customer) > 1}),
    }

"""Common route construction for EDD, greedy, GP and GPRT; simulator independent."""
from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from typing import Callable

from vrptw_baselines.data import VRPTWInstance
from vrptw_baselines.solver import CandidateContext, _candidate_contexts


def visible_problem(instance: VRPTWInstance, released_ids: set[int]) -> tuple[VRPTWInstance, dict[int, int]]:
    """Expose only revealed data, including for normalization and look-ahead."""
    ids = [0, *sorted(released_ids)]
    to_local = {original: local for local, original in enumerate(ids)}
    snapshot = replace(
        instance,
        customers=tuple(replace(instance.customers[original], customer_id=local)
                        for local, original in enumerate(ids)),
        sha256=hashlib.sha256(f"{instance.sha256}:visible:{ids}".encode()).hexdigest(),
    )
    return snapshot, to_local


def construct_route(instance: VRPTWInstance, waiting: set[int], time_now: float,
                    routes_used: int, policy_name: str,
                    priority: Callable[[VRPTWInstance, CandidateContext], float] | None) -> list[int]:
    """Minimize the learned priority after the common hard feasibility filter."""
    current_id = 0
    current_time = max(time_now, instance.depot.ready_time)
    remaining_capacity = instance.capacity
    remaining = set(waiting)
    route: list[int] = []
    while remaining:
        contexts = _candidate_contexts(instance, remaining, current_id, current_time,
                                       remaining_capacity, routes_used)
        if not contexts:
            break
        if priority is not None:
            prepare = getattr(priority, "prepare_contexts", None)
            if prepare is not None:
                prepare(instance, contexts)

        def score(context: CandidateContext) -> tuple[float, int]:
            if priority is not None:
                value = priority(instance, context)
                if not math.isfinite(value):
                    value = math.copysign(1e12, value if not math.isnan(value) else 1.0)
                value = max(-1e12, min(1e12, value))
            elif policy_name == "greedy":
                value = instance.distance(context.current_id, context.customer_id)
            else:
                value = instance.customers[context.customer_id].due_date
            return value, context.customer_id

        chosen = min(contexts, key=score)
        customer = instance.customers[chosen.customer_id]
        route.append(chosen.customer_id)
        remaining.remove(chosen.customer_id)
        remaining_capacity -= customer.demand
        current_time = chosen.service_start + customer.service_time
        current_id = chosen.customer_id
    return route

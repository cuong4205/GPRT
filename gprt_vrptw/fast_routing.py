"""Optional CPU kernel for the existing constructive policy, not a new heuristic.

All trees share one compiled postfix interpreter. No per-individual JIT compilation,
fast-math, float32, candidate pruning, or changes to the route feasibility rules.
"""
from __future__ import annotations

import math
from collections import OrderedDict
from functools import lru_cache

import numpy as np

from vrptw_baselines.data import VRPTWInstance, _distance_matrix
from .grammar import ARITY, FEATURES, Tree

try:
    from numba import njit
except ImportError:
    njit = None


def available() -> bool:
    return njit is not None


_OPS = {"add": 100, "sub": 101, "mul": 102, "pdiv": 103, "min": 104,
        "max": 105, "ge": 106, "le": 107, "and": 108, "or": 109, "if_else": 110}
_FEATURES_V2 = ("d_current", "d_depot", "demand", "ready_time", "due_date", "service_time",
                "arrival_time", "slack", "wait_time", "remaining_capacity", "route_load",
                "remaining_fraction", "current_time", "remaining_vehicles",
                "next_feasible_fraction", "urgency_rank")


@lru_cache(maxsize=8192)
def pack_program(tree: Tree) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if FEATURES != _FEATURES_V2 or set(ARITY) != set(_OPS):
        raise ValueError("Native routing kernel requires the project's feature-v2 grammar")
    codes: list[int] = []
    constants: list[float] = []
    used = np.zeros(len(FEATURES), dtype=np.bool_)

    def visit(node: Tree) -> None:
        if isinstance(node, str):
            index = FEATURES.index(node)
            used[index] = True
            codes.append(index)
            constants.append(0.0)
        elif isinstance(node, (int, float)):
            codes.append(-1)
            constants.append(float(node))
        else:
            op = str(node[0])
            if op not in ARITY or len(node) != ARITY[op] + 1:
                raise ValueError(f"Invalid operator node: {node}")
            for child in node[1:]:
                visit(child)
            codes.append(_OPS[op])
            constants.append(0.0)

    visit(tree)
    result = np.asarray(codes, dtype=np.int64), np.asarray(constants, dtype=np.float64), used
    for array in result:
        array.setflags(write=False)
    return result


# Key by object identity, retaining the object to prevent ID reuse. Bound the
# extra dense matrices just like the canonical distance cache (about 8 MB).
_INPUTS: OrderedDict[int, tuple[VRPTWInstance, np.ndarray, np.ndarray]] = OrderedDict()
_INPUT_ENTRIES = 0
_MAX_INPUT_ENTRIES = 1_000_000


def _inputs(instance: VRPTWInstance) -> tuple[np.ndarray, np.ndarray] | None:
    global _INPUT_ENTRIES
    key = id(instance)
    cached = _INPUTS.get(key)
    if cached is not None and cached[0] is instance:
        _INPUTS.move_to_end(key)
        return cached[1], cached[2]
    matrix = _distance_matrix(instance)
    if matrix is None:
        return None
    customers = np.asarray([(c.demand, c.ready_time, c.due_date, c.service_time)
                            for c in instance.customers], dtype=np.float64)
    distances = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(customers).all() or not np.isfinite(distances).all():
        return None  # Preserve the Python API's handling of noncanonical inputs.
    entries = distances.size
    while _INPUTS and (_INPUT_ENTRIES + entries > _MAX_INPUT_ENTRIES or len(_INPUTS) >= 64):
        _, (_, _, old_distances) = _INPUTS.popitem(last=False)
        _INPUT_ENTRIES -= old_distances.size
    customers.setflags(write=False)
    distances.setflags(write=False)
    _INPUTS[key] = instance, customers, distances
    _INPUT_ENTRIES += entries
    return customers, distances


def _bounded(value: float) -> float:
    if not math.isfinite(value):
        return math.copysign(1e6, value if not math.isnan(value) else 1.0)
    return max(-1e6, min(1e6, value))


def _program_score(codes, constants, features, stack):
    top = 0
    for index in range(len(codes)):
        code = codes[index]
        if code < 100:
            stack[top] = constants[index] if code == -1 else features[code]
            top += 1
            continue
        if code == 110:  # if_else; arguments are evaluated eagerly by compile_tree.
            top -= 2
            stack[top - 1] = stack[top] if bool(stack[top - 1]) else stack[top + 1]
            continue
        top -= 1
        a, b = stack[top - 1], stack[top]
        if code == 100:
            value = _bounded(a + b)
        elif code == 101:
            value = _bounded(a - b)
        elif code == 102:
            value = _bounded(a * b)
        elif code == 103:
            value = _bounded(a if abs(b) < 1e-9 else a / b)
        elif code == 104:
            value = b if b < a else a  # Match Python min/max's operand order.
        elif code == 105:
            value = b if b > a else a
        elif code == 106:
            value = 1.0 if a >= b else 0.0
        elif code == 107:
            value = 1.0 if a <= b else 0.0
        elif code == 108:
            value = 1.0 if bool(a) and bool(b) else 0.0
        else:  # or
            value = 1.0 if bool(a) or bool(b) else 0.0
        stack[top - 1] = value
    return _bounded(stack[0])


def _construct(customers, distances, capacity, max_vehicles, scale, codes, constants, used):
    count = len(customers)
    horizon = max(1.0, customers[0, 2])
    capacity_scale = max(1.0, capacity)
    customer_count = max(1, count - 1)
    unserved = np.ones(count, dtype=np.bool_)
    unserved[0] = False
    remaining = count - 1
    routes = np.empty((min(max_vehicles, remaining), count + 2), dtype=np.int64)
    lengths = np.zeros(len(routes), dtype=np.int64)
    features = np.empty(len(used), dtype=np.float64)
    stack = np.empty(len(codes), dtype=np.float64)
    route_count = 0
    while remaining and route_count < max_vehicles:
        current_id = 0
        current_time = max(0.0, customers[0, 1])
        remaining_capacity = capacity
        position = 1
        routes[route_count, 0] = 0
        while remaining:
            chosen = -1
            best_score = math.inf
            chosen_start = 0.0
            for cid in range(1, count):
                if not unserved[cid] or customers[cid, 0] > remaining_capacity + 1e-9:
                    continue
                arrival = current_time + distances[current_id, cid]
                service_start = max(arrival, customers[cid, 1])
                if service_start > customers[cid, 2] + 1e-9:
                    continue
                departure = service_start + customers[cid, 3]
                if departure + distances[cid, 0] > customers[0, 2] + 1e-9:
                    continue
                if used[0]: features[0] = distances[current_id, cid] / scale
                if used[1]: features[1] = distances[cid, 0] / scale
                if used[2]: features[2] = customers[cid, 0] / capacity_scale
                if used[3]: features[3] = customers[cid, 1] / horizon
                if used[4]: features[4] = customers[cid, 2] / horizon
                if used[5]: features[5] = customers[cid, 3] / horizon
                if used[6]: features[6] = arrival / horizon
                if used[7]: features[7] = (customers[cid, 2] - arrival) / horizon
                if used[8]: features[8] = max(0.0, customers[cid, 1] - arrival) / horizon
                if used[9]: features[9] = remaining_capacity / capacity_scale
                if used[10]: features[10] = (capacity - remaining_capacity) / capacity_scale
                if used[11]: features[11] = remaining / customer_count
                if used[12]: features[12] = current_time / horizon
                if used[13]: features[13] = max(0, max_vehicles - route_count - 1) / max(1, max_vehicles)
                if used[14]:
                    next_feasible = 0
                    capacity_after = remaining_capacity - customers[cid, 0]
                    for other in range(1, count):
                        if not unserved[other] or other == cid:
                            continue
                        if customers[other, 0] > capacity_after + 1e-9:
                            continue
                        next_arrival = departure + distances[cid, other]
                        next_start = max(next_arrival, customers[other, 1])
                        if next_start > customers[other, 2] + 1e-9:
                            continue
                        if next_start + customers[other, 3] + distances[other, 0] > customers[0, 2] + 1e-9:
                            continue
                        next_feasible += 1
                    features[14] = next_feasible / max(1, remaining - 1)
                if used[15]:
                    rank = 0
                    for other in range(1, count):
                        if unserved[other] and customers[other, 2] <= customers[cid, 2]:
                            rank += 1
                    features[15] = rank / max(1, remaining)
                score = _program_score(codes, constants, features, stack)
                # Candidates are visited in ascending ID, including exact ties.
                if chosen == -1 or score < best_score:
                    chosen, best_score, chosen_start = cid, score, service_start
            if chosen == -1:
                break
            routes[route_count, position] = chosen
            position += 1
            unserved[chosen] = False
            remaining -= 1
            remaining_capacity -= customers[chosen, 0]
            current_time = chosen_start + customers[chosen, 3]
            current_id = chosen
        if position == 1:
            break
        routes[route_count, position] = 0
        lengths[route_count] = position + 1
        route_count += 1
    return routes, lengths, route_count


if njit is not None:
    # Keep helpers in this file so the disk cache invalidates with kernel changes.
    _bounded = njit(cache=True, fastmath=False)(_bounded)
    _program_score = njit(cache=True, fastmath=False)(_program_score)
    _construct = njit(cache=True, fastmath=False, nogil=True)(_construct)


def construct_routes(instance: VRPTWInstance, program, scale: float) -> list[tuple[int, ...]] | None:
    if not available():
        raise RuntimeError("Numba backend unavailable; install requirements-fast.txt")
    inputs = _inputs(instance)
    if inputs is None:
        return None
    customers, distances = inputs
    routes, lengths, count = _construct(customers, distances, float(instance.capacity),
                                        instance.max_vehicles, float(scale), *program)
    return [tuple(int(cid) for cid in routes[index, :lengths[index]]) for index in range(count)]

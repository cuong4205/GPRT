from __future__ import annotations

import hashlib
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Customer:
    customer_id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_date: float
    service_time: float


@dataclass(frozen=True, slots=True)
class VRPTWInstance:
    name: str
    path: Path
    max_vehicles: int
    capacity: float
    customers: tuple[Customer, ...]
    sha256: str
    source: str
    _distance_matrix_cache: tuple[tuple[float, ...], ...] | None = field(
        default=None, init=False, repr=False, compare=False, hash=False
    )

    @property
    def depot(self) -> Customer:
        return self.customers[0]

    @property
    def customer_count(self) -> int:
        return len(self.customers) - 1

    def distance(self, from_id: int, to_id: int) -> float:
        matrix = self._distance_matrix_cache
        if matrix is None:
            matrix = _distance_matrix(self)
        if matrix is not None:
            return matrix[from_id][to_id]
        a = self.customers[from_id]
        b = self.customers[to_id]
        return math.hypot(a.x - b.x, a.y - b.y)


# Keep cached matrices within a bounded number of pair values. A standard
# Solomon instance needs about 10,000 entries; oversized instances use the
# original calculation instead of evicting the whole cache on every call.
_MAX_CACHED_DISTANCE_ENTRIES = 1_000_000
_DISTANCE_MATRICES: OrderedDict[
    int, tuple[VRPTWInstance, tuple[tuple[float, ...], ...], int]
] = OrderedDict()
_CACHED_DISTANCE_ENTRIES = 0
_DISTANCE_MATRIX_BUILDS = 0


def _distance_matrix(
    instance: VRPTWInstance,
) -> tuple[tuple[float, ...], ...] | None:
    """Return a bounded, per-instance matrix of exact math.hypot distances."""
    global _CACHED_DISTANCE_ENTRIES, _DISTANCE_MATRIX_BUILDS

    key = id(instance)
    cached = _DISTANCE_MATRICES.get(key)
    if cached is not None and cached[0] is instance:
        _DISTANCE_MATRICES.move_to_end(key)
        return cached[1]

    customer_count = len(instance.customers)
    entry_count = customer_count * customer_count
    if entry_count > _MAX_CACHED_DISTANCE_ENTRIES:
        return None

    _DISTANCE_MATRIX_BUILDS += 1
    matrix = tuple(
        tuple(
            math.hypot(a.x - b.x, a.y - b.y)
            for b in instance.customers
        )
        for a in instance.customers
    )
    while (
        _DISTANCE_MATRICES
        and _CACHED_DISTANCE_ENTRIES + entry_count > _MAX_CACHED_DISTANCE_ENTRIES
    ):
        _, (
            evicted_instance,
            evicted_matrix,
            evicted_entries,
        ) = _DISTANCE_MATRICES.popitem(last=False)
        if evicted_instance._distance_matrix_cache is evicted_matrix:
            object.__setattr__(evicted_instance, "_distance_matrix_cache", None)
        _CACHED_DISTANCE_ENTRIES -= evicted_entries
    _DISTANCE_MATRICES[key] = (instance, matrix, entry_count)
    _CACHED_DISTANCE_ENTRIES += entry_count
    object.__setattr__(instance, "_distance_matrix_cache", matrix)
    return matrix


def distance_cache_info() -> dict[str, int]:
    """Return size and build counters for the bounded distance-matrix cache."""
    return {
        "matrix_builds": _DISTANCE_MATRIX_BUILDS,
        "instances": len(_DISTANCE_MATRICES),
        "entries": _CACHED_DISTANCE_ENTRIES,
        "max_entries": _MAX_CACHED_DISTANCE_ENTRIES,
    }


_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _numbers(line: str) -> list[float]:
    return [float(value) for value in _NUMBER_RE.findall(line)]


def parse_instance(path: str | Path) -> VRPTWInstance:
    """Parse Solomon and Gehring-Homberger text instances.

    Both datasets use the same seven-column customer layout. Customer identifiers
    must be contiguous and start at depot 0 so that route checking is unambiguous.
    """

    path = Path(path)
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    nonempty = [line.strip() for line in lines if line.strip()]
    if not nonempty:
        raise ValueError(f"Empty VRPTW instance: {path}")

    name = nonempty[0]
    vehicle_idx = next(
        (i for i, line in enumerate(lines) if line.strip().upper() == "VEHICLE"), None
    )
    customer_idx = next(
        (i for i, line in enumerate(lines) if line.strip().upper() == "CUSTOMER"), None
    )
    if vehicle_idx is None or customer_idx is None:
        raise ValueError(f"Missing VEHICLE/CUSTOMER section: {path}")

    vehicle_values: list[float] | None = None
    for line in lines[vehicle_idx + 1 : customer_idx]:
        values = _numbers(line)
        if len(values) == 2:
            vehicle_values = values
            break
    if vehicle_values is None:
        raise ValueError(f"Missing vehicle count and capacity: {path}")

    customers: list[Customer] = []
    for line in lines[customer_idx + 1 :]:
        values = _numbers(line)
        if len(values) != 7:
            continue
        customer_id, x, y, demand, ready, due, service = values
        customers.append(
            Customer(int(customer_id), x, y, demand, ready, due, service)
        )
    if len(customers) < 2:
        raise ValueError(f"No customer records found: {path}")
    ids = [customer.customer_id for customer in customers]
    if ids != list(range(len(customers))):
        raise ValueError(f"Customer IDs must be contiguous from zero: {path}")

    source = "Gehring-Homberger" if "Gehring_Homberger" in path.parts else "Solomon"
    return VRPTWInstance(
        name=name,
        path=path.resolve(),
        max_vehicles=int(vehicle_values[0]),
        capacity=vehicle_values[1],
        customers=tuple(customers),
        sha256=hashlib.sha256(raw).hexdigest(),
        source=source,
    )


def discover_instances(data_dir: str | Path) -> list[VRPTWInstance]:
    paths = sorted(
        (path for path in Path(data_dir).rglob("*") if path.suffix.lower() == ".txt"),
        key=lambda path: str(path).lower(),
    )
    return [parse_instance(path) for path in paths]

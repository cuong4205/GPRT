from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
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

    @property
    def depot(self) -> Customer:
        return self.customers[0]

    @property
    def customer_count(self) -> int:
        return len(self.customers) - 1

    def distance(self, from_id: int, to_id: int) -> float:
        a = self.customers[from_id]
        b = self.customers[to_id]
        return math.hypot(a.x - b.x, a.y - b.y)


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

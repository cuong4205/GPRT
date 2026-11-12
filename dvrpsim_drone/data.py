"""CSV benchmark inputs and reproducible layout holdouts."""
from __future__ import annotations

import csv
import hashlib
import io
import math
from dataclasses import dataclass
from functools import reduce
from operator import truediv
from pathlib import Path

from vrptw_baselines.data import Customer, VRPTWInstance

COLUMNS = ("x", "y", "demand", "open", "close", "servicetime", "drone_serve", "time")


@dataclass(frozen=True)
class Settings:
    truck_speed: float
    drone_speed: float
    truck_capacity: float
    drone_capacity: float
    truck_service: float
    drone_service: float
    service_window: float
    endurance: float
    launch: float
    recovery: float

    @classmethod
    def read(cls, path: Path) -> Settings:
        values = dict(line.split()[:2] for line in path.read_text().splitlines() if line.strip())
        keys = ("TruckSpeed", "DroneSpeed", "TruckCapacity", "DroneCapacity",
                "ServiceTimeTruck", "ServiceTimeDrone", "ServiceWindow", "Endurance",
                "LaunchTime", "RecoveryTime")
        numbers = [reduce(truediv, map(float, values[key].split("/"))) for key in keys]
        if any(not math.isfinite(value) or value <= 0 for value in numbers):
            raise ValueError(f"Invalid settings: {path}")
        return cls(*numbers)


@dataclass(frozen=True, slots=True)
class DroneInstance(VRPTWInstance):
    release: tuple[float, ...]
    drone_allowed: tuple[bool, ...]
    layout: str


def load_instance(path: Path, settings: Settings, fleet: int) -> DroneInstance:
    if fleet < 1:
        raise ValueError("Fleet size must be positive")
    payload = path.read_bytes()
    reader = csv.reader(io.StringIO(payload.decode("utf-8-sig")))
    if tuple(next(reader, ())) != COLUMNS:
        raise ValueError(f"Unexpected CSV header: {path}")
    customers, releases, allowed = [], [], []
    for cid, row in enumerate(reader):
        x, y, demand, opening, closing, service, flag, release = map(float, row)
        if (not all(map(math.isfinite, (x, y, demand, opening, closing, service, flag, release)))
                or not 0 <= release <= opening <= closing or service < 0 or flag not in (0, 1)
                or (cid == 0 and (demand != 0 or opening != 0))
                or (cid > 0 and not 0 < demand <= settings.truck_capacity)):
            raise ValueError(f"Invalid row {cid + 2}: {path}")
        customers.append(Customer(cid, x, y, demand, opening, closing, service))
        releases.append(release)
        allowed.append(bool(flag) and demand <= settings.drone_capacity and cid != 0)
    if len(customers) < 2:
        raise ValueError(f"Missing depot or customers: {path}")
    layout = sorted((c.x, c.y) for c in customers[1:])
    return DroneInstance(path.stem.lower(), path, fleet, settings.truck_capacity,
                         tuple(customers), hashlib.sha256(payload).hexdigest(), "dvrptw-drone",
                         tuple(releases), tuple(allowed),
                         hashlib.sha256(repr(layout).encode()).hexdigest())


def load_dataset(root: Path, settings: Settings, fleet: int) -> list[DroneInstance]:
    instances = []
    for size in (100, 200, 400):
        files = sorted((root / str(size)).glob("*.csv"))
        if not files:
            raise ValueError(f"Missing CSV instances for size {size}: {root}")
        for path in files:
            instance = load_instance(path, settings, fleet)
            if instance.customer_count != size:
                raise ValueError(f"Expected {size} customers: {path}")
            instances.append(instance)
    if len({item.name for item in instances}) != len(instances):
        raise ValueError("Duplicate instance names")
    return instances


def layout_split(instances: list[DroneInstance], fold: int) -> dict[str, str]:
    layouts = sorted({item.layout for item in instances if item.customer_count == 100})
    if len(layouts) != 4 or not 0 <= fold < len(layouts):
        raise ValueError("Expected four 100-customer layouts and fold in [0, 3]")
    test, validation = layouts[fold], layouts[(fold + 1) % len(layouts)]
    return {
        item.name: ("transfer" if item.customer_count != 100 else "test" if item.layout == test
                    else "validation" if item.layout == validation else "train")
        for item in instances
    }

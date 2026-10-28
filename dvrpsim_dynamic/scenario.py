"""Materialized, policy-independent Solomon dynamic scenarios (no simulator API)."""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

from vrptw_baselines.data import VRPTWInstance, validate_instance


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    dynamic_ratio: float = 0.70
    dynamic_lead_min: int = 10
    dynamic_lead_max: int = 15
    cancel_candidate_ratio: float = 0.05
    cancel_delay_min: int = 1
    cancel_delay_max: int = 15

    def __post_init__(self) -> None:
        for name in ("dynamic_ratio", "cancel_candidate_ratio"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for low, high in ((self.dynamic_lead_min, self.dynamic_lead_max),
                          (self.cancel_delay_min, self.cancel_delay_max)):
            if type(low) is not int or type(high) is not int or not 0 <= low <= high:
                raise ValueError("Lead/delay bounds must be nonnegative integers with min <= max")


@dataclass(frozen=True, slots=True)
class ScenarioOrder:
    customer_id: int
    release_time: float
    dynamic_selected: bool = False
    lead_time: int | None = None
    cancel_time: float | None = None


@dataclass(frozen=True, slots=True)
class Scenario:
    instance: str
    instance_sha256: str
    seed: int | None
    mode: str
    rules: ScenarioConfig
    orders: tuple[ScenarioOrder, ...]

    @property
    def release_times(self) -> dict[int, float]:
        return {order.customer_id: order.release_time for order in self.orders}

    @property
    def cancellation_requests(self) -> tuple[tuple[int, float], ...]:
        return tuple((order.customer_id, order.cancel_time) for order in self.orders
                     if order.cancel_time is not None)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "rounding": "decimal_half_up",
                "random_streams": "sha256(seed:purpose), python.Random",
                **asdict(self)}

    @property
    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                         separators=(",", ":"), allow_nan=False).encode()).hexdigest()

    def validate(self, instance: VRPTWInstance) -> None:
        validate_instance(instance)
        if self.instance.lower() != instance.name.lower() or self.instance_sha256 != instance.sha256:
            raise ValueError("Scenario does not match the instance name/content hash")
        if self.mode not in {"static", "dynamic"}:
            raise ValueError("Scenario mode must be static or dynamic")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("Scenario seed must be an integer")
        if [o.customer_id for o in self.orders] != list(range(1, len(instance.customers))):
            raise ValueError("Scenario must contain each customer exactly once, in ID order")
        for order in self.orders:
            if type(order.customer_id) is not int or type(order.dynamic_selected) is not bool:
                raise ValueError("Scenario customer IDs must be integers and dynamic flags boolean")
            for value in (order.release_time, order.cancel_time):
                if value is not None and (not math.isfinite(value) or value < 0):
                    raise ValueError("Scenario event times must be finite and nonnegative")
            if self.mode == "static" and (order.release_time != 0 or order.cancel_time is not None):
                raise ValueError("Static control must release all orders at zero without cancellations")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2,
                                   allow_nan=False) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, instance: VRPTWInstance) -> Scenario:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.pop("schema_version", None) != 1:
            raise ValueError("Unsupported scenario schema_version")
        if payload.pop("rounding", None) != "decimal_half_up":
            raise ValueError("Unsupported scenario rounding rule")
        if payload.pop("random_streams", None) != "sha256(seed:purpose), python.Random":
            raise ValueError("Unsupported scenario random streams")
        scenario = cls(**{**payload, "rules": ScenarioConfig(**payload["rules"]),
                         "orders": tuple(ScenarioOrder(**row) for row in payload["orders"])})
        scenario.validate(instance)
        return scenario


def _count(count: int, ratio: float) -> int:
    return int((Decimal(count) * Decimal(str(ratio))).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _rng(seed: int, purpose: str) -> random.Random:
    return random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{purpose}".encode()).digest(), "big"))


def generate_scenario(instance: VRPTWInstance, seed: int = 42, *,
                      rules: ScenarioConfig | None = None, mode: str = "dynamic") -> Scenario:
    validate_instance(instance)
    if type(seed) is not int:
        raise ValueError("Scenario seed must be an integer")
    rules = rules or ScenarioConfig()
    ids = list(range(1, len(instance.customers)))
    dynamic = set(_rng(seed, "dynamic_selection").sample(ids, _count(len(ids), rules.dynamic_ratio)))
    cancellations = set(_rng(seed, "cancel_selection").sample(ids, _count(len(ids), rules.cancel_candidate_ratio)))
    lead_rng, delay_rng = _rng(seed, "lead_time"), _rng(seed, "cancel_delay")
    orders = []
    for customer_id in ids:
        selected = mode == "dynamic" and customer_id in dynamic
        lead = lead_rng.randint(rules.dynamic_lead_min, rules.dynamic_lead_max) if selected else None
        release = float(max(0, instance.customers[customer_id].ready_time - lead)) if selected else 0.0
        cancel = (release + delay_rng.randint(rules.cancel_delay_min, rules.cancel_delay_max)
                  if mode == "dynamic" and customer_id in cancellations else None)
        orders.append(ScenarioOrder(customer_id, release, selected, lead, cancel))
    scenario = Scenario(instance.name, instance.sha256, seed if mode == "dynamic" else None,
                        mode, rules, tuple(orders))
    scenario.validate(instance)
    return scenario

from __future__ import annotations

import hashlib
import logging
import math
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from importlib.metadata import version

from dvrpsim import Location, Model, Order, Vehicle, VehicleLoading
from dvrpsim.utils.logging import LoggingCallback

from gprt_vrptw.evaluator import make_priority
from gprt_vrptw.grammar import ARITY, FEATURES, Tree, tree_to_prefix, tree_to_string
from vrptw_baselines.data import VRPTWInstance
from .routing import construct_route, visible_problem
from .scenario import Scenario, ScenarioConfig, ScenarioOrder


ScenarioMode = Literal["static", "dynamic"]


class _QuietLoggingCallback(LoggingCallback):
    """Keep per-event simulator logs out of stdout and the project directory."""

    def __init__(self, model: Model) -> None:
        self.model = model
        self.logger = logging.getLogger("dvrpsim-dynamic-quiet")
        self.logger.disabled = True
        self.config: dict[str, Any] = {}

    def _configure_logger(self) -> None:
        return


class DynamicEpisodeError(RuntimeError):
    """An episode could not be simulated without changing its stated scenario."""


@dataclass(frozen=True, slots=True)
class PolicySpec:
    name: str
    training_seed: int | None
    expression: str
    expression_sha256: str
    source_file: str
    source_sha256: str
    checkpoint_sha256: str | None
    tree: Tree | None


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_expression(expression: str) -> Tree:
    """Parse the project's printed prefix-tree form without evaluating code."""
    tokens: list[str] = []
    cursor = 0
    token_pattern = re.compile(r"\s*([A-Za-z_]\w*|[-+]?(?:\d+(?:\.\d*)?|\.\d+)|[(),])")
    while cursor < len(expression):
        match = token_pattern.match(expression, cursor)
        if match is None:
            raise ValueError(f"Invalid character at position {cursor} in heuristic expression")
        tokens.append(match.group(1))
        cursor = match.end()
    position = 0

    def take(expected: str | None = None) -> str:
        nonlocal position
        if position >= len(tokens):
            raise ValueError("Unexpected end of heuristic expression")
        token = tokens[position]
        if expected is not None and token != expected:
            raise ValueError(f"Expected {expected!r}, found {token!r}")
        position += 1
        return token

    def parse() -> Tree:
        token = take()
        if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", token):
            return float(token)
        if token in FEATURES:
            return token
        if token not in ARITY:
            raise ValueError(f"Unknown heuristic feature or operator: {token}")
        take("(")
        children = [parse()]
        while position < len(tokens) and tokens[position] == ",":
            take(",")
            children.append(parse())
        take(")")
        if len(children) != ARITY[token]:
            raise ValueError(
                f"Operator {token} expects {ARITY[token]} arguments, found {len(children)}"
            )
        return (token, *children)

    tree = parse()
    if position != len(tokens):
        raise ValueError(f"Trailing token after complete expression: {tokens[position]}")
    tree_to_prefix(tree)  # Enforce the exact vocabulary and constant set.
    if tree_to_string(tree) != expression:
        raise ValueError("Expression is valid but is not in canonical project format")
    return tree


def load_selected_policies(
    project_root: str | Path,
    methods: tuple[str, ...] = ("gprt", "gp"),
    *,
    policy_root: str | Path = "outputs/gprt_vrptw/reduced_main",
    training_seeds: tuple[int, ...] | None = None,
) -> list[PolicySpec]:
    """Load validation-selected expressions for every completed training seed."""

    root = Path(project_root)
    policies: list[PolicySpec] = []
    for method in methods:
        if method not in {"gp", "gprt"}:
            raise ValueError(f"Unsupported learned policy: {method}")
        method_dir = root / policy_root / method
        if not method_dir.is_dir():
            raise DynamicEpisodeError(f"Missing policy directory: {method_dir}")
        for seed_dir in sorted(path for path in method_dir.iterdir() if path.is_dir() and path.name.isdecimal()):
            if training_seeds is not None and int(seed_dir.name) not in training_seeds:
                continue
            selected_path = seed_dir / "final_heuristic.txt"
            if not selected_path.is_file():
                continue
            raw = selected_path.read_bytes()
            expression = raw.decode("utf-8").strip()
            tree = parse_expression(expression)
            checkpoint = seed_dir / "best.pt"
            policies.append(
                PolicySpec(
                    name=method,
                    training_seed=int(seed_dir.name),
                    expression=expression,
                    expression_sha256=hashlib.sha256(raw).hexdigest(),
                    source_file=str(selected_path.resolve()),
                    source_sha256=sha256_file(selected_path),
                    checkpoint_sha256=sha256_file(checkpoint) if checkpoint.is_file() else None,
                    tree=tree,
                )
            )

    for method in methods:
        found = {item.training_seed for item in policies if item.name == method}
        if not found or (training_seeds is not None and found != set(training_seeds)):
            raise DynamicEpisodeError(f"Missing validation-selected {method} artifacts for requested seeds")
    return policies


def make_release_times(
    instance: VRPTWInstance,
    mode: ScenarioMode,
    scenario_seed: int | None,
    *,
    release_window_fraction: float = 0.5,
) -> dict[int, float]:
    """Legacy uniform-release scenario; new experiments use generate_scenario."""
    if mode == "static":
        return {customer_id: 0.0 for customer_id in range(1, len(instance.customers))}
    if scenario_seed is None:
        raise ValueError("Dynamic scenarios require a scenario seed")
    if not 0 < release_window_fraction <= 1:
        raise ValueError("release_window_fraction must be in (0, 1]")

    import random

    rng = random.Random(scenario_seed)
    horizon = instance.depot.due_date
    releases: dict[int, float] = {}
    for customer_id, customer in enumerate(instance.customers[1:], start=1):
        outbound = instance.distance(0, customer_id)
        return_leg = instance.distance(customer_id, 0)
        earliest_direct_start = max(outbound, customer.ready_time)
        if earliest_direct_start > customer.due_date or (
            earliest_direct_start + customer.service_time + return_leg > horizon
        ):
            raise DynamicEpisodeError(
                "Cannot construct a direct-service-feasible release scenario for "
                f"{instance.name}: customer {customer_id} is infeasible even at "
                "release time 0 under Solomon's latest-service-start semantics."
            )
        # Solomon's due_date is a latest service-start bound, so service duration
        # belongs in the return-to-depot horizon, not in the customer due bound.
        latest_individual_release = min(
            customer.due_date - outbound,
            horizon - outbound - customer.service_time - return_leg,
        )
        if latest_individual_release < 0:
            raise DynamicEpisodeError(
                "Cannot construct the requested direct-service-feasible release "
                f"scenario for {instance.name}: customer {customer_id} has a "
                f"negative release bound ({latest_individual_release:.12g})."
            )
        proposed = rng.uniform(0.0, release_window_fraction * horizon)
        releases[customer_id] = min(proposed, latest_individual_release)
    return releases


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def audit_routes(
    instance: VRPTWInstance,
    routes: list[dict[str, Any]],
    service_starts: dict[int, float],
    delivered_customer_ids: set[int],
    *,
    tolerance: float = 1e-6,
    release_times: dict[int, float] | None = None,
    cancelled_customer_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Independently check capacity, time windows, coverage, and simulator timing."""
    violations: list[dict[str, Any]] = []
    planned: list[int] = []
    if len({route.get("vehicle_id", index) for index, route in enumerate(routes)}) > instance.max_vehicles:
        violations.append({"type": "fleet_size"})
    for route_index, route in enumerate(routes):
        customer_ids = [int(value) for value in route["customer_ids"]]
        planned.extend(customer_ids)
        total_demand = sum(instance.customers[value].demand for value in customer_ids)
        if total_demand > instance.capacity + tolerance:
            violations.append(
                {
                    "type": "capacity",
                    "route_index": route_index,
                    "load": total_demand,
                    "capacity": instance.capacity,
                }
            )

        current_id = 0
        current_time = float(route["dispatch_time"])
        if current_time < instance.depot.ready_time - tolerance:
            violations.append({"type": "depot_ready_time", "route_index": route_index})
        for customer_id in customer_ids:
            if release_times is not None and release_times[customer_id] > float(route.get("assignment_time", route["dispatch_time"])) + tolerance:
                violations.append({"type": "information_leakage", "customer_id": customer_id})
            if cancelled_customer_ids is not None and customer_id in cancelled_customer_ids:
                violations.append({"type": "cancelled_order_assigned", "customer_id": customer_id})
            customer = instance.customers[customer_id]
            arrival = current_time + instance.distance(current_id, customer_id)
            service_start = max(arrival, customer.ready_time)
            actual_start = service_starts.get(customer_id)
            if actual_start is None:
                violations.append(
                    {"type": "planned_order_not_served", "customer_id": customer_id}
                )
            else:
                if abs(actual_start - service_start) > tolerance:
                    violations.append(
                        {
                            "type": "simulator_timing_mismatch",
                            "customer_id": customer_id,
                            "expected_start": service_start,
                            "actual_start": actual_start,
                        }
                    )
                if actual_start < customer.ready_time - tolerance:
                    violations.append(
                        {
                            "type": "ready_time",
                            "customer_id": customer_id,
                            "service_start": actual_start,
                            "ready_time": customer.ready_time,
                        }
                    )
                if actual_start > customer.due_date + tolerance:
                    violations.append(
                        {
                            "type": "due_date",
                            "customer_id": customer_id,
                            "service_start": actual_start,
                            "due_date": customer.due_date,
                        }
                    )
            current_time = service_start + customer.service_time
            current_id = customer_id

        returned = current_time + instance.distance(current_id, 0)
        if customer_ids and returned > instance.depot.due_date + tolerance:
            violations.append(
                {
                    "type": "depot_horizon",
                    "route_index": route_index,
                    "return_time": returned,
                    "depot_due_date": instance.depot.due_date,
                }
            )
        if "return_time" in route and abs(route["return_time"] - returned) > tolerance:
            violations.append({"type": "return_timing_mismatch", "route_index": route_index})

    duplicates = sorted({customer_id for customer_id in planned if planned.count(customer_id) > 1})
    for customer_id in duplicates:
        violations.append({"type": "duplicate_assignment", "customer_id": customer_id})
    for customer_id in sorted(delivered_customer_ids - set(planned)):
        violations.append({"type": "unplanned_delivery", "customer_id": customer_id})
    return violations


def run_episode(
    instance: VRPTWInstance,
    release_times: dict[int, float],
    policy_name: str,
    *,
    policy: PolicySpec | None = None,
    scenario_mode: ScenarioMode,
    scenario_seed: int | None,
    episode_id: str,
    capture_events: bool = True,
    scenario: Scenario | None = None,
    cancellation_requests: tuple[tuple[int, float], ...] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one Solomon episode in dvrpsim with a shared dynamic dispatcher."""
    if version("dvrpsim") != "1.0.0":
        raise DynamicEpisodeError("This adapter requires dvrpsim==1.0.0")
    if version("simpy") != "4.1.1":
        raise DynamicEpisodeError("This adapter requires simpy==4.1.1")
    if policy_name not in {"gprt", "gp", "edd", "greedy"}:
        raise ValueError(f"Unsupported dynamic policy: {policy_name}")
    if policy_name in {"gprt", "gp"} and (policy is None or policy.tree is None or policy.name != policy_name):
        raise ValueError(f"Policy artifact is required for {policy_name}")
    if policy_name in {"edd", "greedy"} and policy is not None:
        raise ValueError("Baseline policy cannot receive a learned expression")
    if scenario is None:
        scenario = Scenario(instance.name, instance.sha256, scenario_seed, scenario_mode,
                            ScenarioConfig(), tuple(ScenarioOrder(cid, value)
                                for cid, value in sorted(release_times.items())))
    scenario.validate(instance)
    if scenario.release_times != release_times or scenario.mode != scenario_mode or scenario.seed != scenario_seed:
        raise ValueError("Scenario metadata and episode arguments disagree")
    cancellation_requests = (*scenario.cancellation_requests, *cancellation_requests)
    for customer_id, cancel_time in cancellation_requests:
        if customer_id not in release_times or not math.isfinite(cancel_time) or cancel_time < 0:
            raise ValueError("Invalid cancellation request customer/time")

    started = time.perf_counter()
    coordinate_by_location = {
        f"L{customer.customer_id:03d}": (customer.x, customer.y)
        for customer in instance.customers
    }
    customer_by_order = {f"O{customer_id:03d}": customer_id for customer_id in release_times}
    order_by_customer = {customer_id: order_id for order_id, customer_id in customer_by_order.items()}
    priority = make_priority(policy.tree) if policy is not None and policy.tree is not None else None

    class SolomonVehicle(Vehicle):
        def travel_distance(self, origin: Location, destination: Location) -> float:
            ax, ay = coordinate_by_location[origin.id]
            bx, by = coordinate_by_location[destination.id]
            return math.hypot(ax - bx, ay - by)

        def travel_time(self, origin: Location, destination: Location) -> float:
            return self.travel_distance(origin, destination)

    class SolomonModel(Model):
        def _init_logger(self) -> None:
            self.log = _QuietLoggingCallback(self)

        def __init__(self) -> None:
            super().__init__()
            self.events: list[dict[str, Any]] = []
            self.route_records: list[dict[str, Any]] = []
            self.service_starts: dict[int, float] = {}
            self.delivered_customer_ids: set[int] = set()
            self.rejected_customer_ids: set[int] = set()
            self.callback_latencies_ms: list[float] = []
            self.total_distance = 0.0
            self.departure_count = 0
            self.end_time = 0.0
            self.used_vehicle_ids: set[str] = set()
            self.leakage_violations: list[dict[str, Any]] = []
            self.cancelled_customer_ids: set[int] = set()
            self.released_customer_ids: set[int] = set()
            self.cancellation_records: list[dict[str, Any]] = []
            self.policy = policy
            self.priority = priority
            self.instance = instance
            self.customer_by_order = customer_by_order
            self.order_by_customer = order_by_customer
            self.coordinate_by_location = coordinate_by_location

            for customer in instance.customers:
                location = Location(
                    id=f"L{customer.customer_id:03d}", x=customer.x, y=customer.y
                )
                self.add_location(location)
            depot = self.get_location_by_id("L000")

            for vehicle_index in range(instance.max_vehicles):
                vehicle = SolomonVehicle(f"V{vehicle_index:03d}")
                vehicle.initial_location = depot
                vehicle.capacity = instance.capacity
                vehicle.loading_rule = VehicleLoading.FIFO
                self.add_vehicle(vehicle)

            for customer_id, release_date in sorted(release_times.items()):
                customer = instance.customers[customer_id]
                order = Order(order_by_customer[customer_id], original_id=str(customer_id))
                order.quantity = customer.demand
                order.release_date = release_date
                order.earliest_pickup_start = instance.depot.ready_time
                order.pickup_location = depot
                order.delivery_location = self.get_location_by_id(f"L{customer_id:03d}")
                order.earliest_delivery_start = customer.ready_time
                order.latest_delivery_start = customer.due_date
                order.delivery_duration = customer.service_time
                self.env.process(self._release_order(order))
            for request_index, (customer_id, cancel_time) in enumerate(cancellation_requests):
                self.env.process(self._cancel_request(customer_id, cancel_time, request_index))
            self.env.process(self._horizon())

        def _release_order(self, order: Order):
            # Use the public request API after checking a pre-release tombstone.
            yield self.env.medium_timeout(order.release_date - self.env.now)
            customer_id = self.customer_by_order[order.id]
            if customer_id in self.cancelled_customer_ids:
                self._event("order_release_suppressed", customer_id=customer_id, order_id=order.id)
                return
            self.request_order(order, decision_point_on_request=True)

        def _cancel_request(self, customer_id: int, cancel_time: float, request_index: int):
            # Native vehicle/release events have priority 3; routing waits at 5.
            # A lower numeric priority runs first in SimPy (names are misleading).
            yield self.env.timeout(cancel_time - self.env.now, priority=4)
            order_id = self.order_by_customer[customer_id]
            order = self.get_order_by_id(order_id)
            assigned_vehicle = self.get_state()["open_orders"].get(order_id, {}).get("assigned_vehicle")
            self._event("cancellation_request", customer_id=customer_id,
                        order_id=order_id, request_index=request_index,
                        assigned_vehicle=assigned_vehicle)
            reason = None
            if customer_id in self.cancelled_customer_ids:
                reason = "already_cancelled"
            elif order is not None and order.is_delivered:
                reason = "already_completed"
            elif assigned_vehicle is not None:
                reason = "already_assigned"
            elif order is not None and (order.is_rejected or not order.can_be_rejected_or_canceled):
                reason = "invalid_state"
            accepted = reason is None
            if accepted:
                self.cancelled_customer_ids.add(customer_id)
                if order is not None:
                    order.cancel()
                else:
                    # Unreleased orders are absent from Model. The tombstone is
                    # checked by the provider before any public request can occur.
                    self.request_for_routing()
            record = {"time": float(self.env.now), "customer_id": customer_id,
                      "request_index": request_index, "accepted": accepted,
                      "reason": reason or ("before_release" if order is None else "unassigned"),
                      "assigned_vehicle": assigned_vehicle}
            self.cancellation_records.append(record)
            self._event("cancellation_accepted" if accepted else "cancellation_rejected",
                        **{key: value for key, value in record.items() if key != "time"})

        def _horizon(self):
            yield self.env.timeout(instance.depot.due_date, priority=6)
            self._event("horizon_reached", unserved_customer_ids=sorted(
                set(release_times) - self.delivered_customer_ids - self.cancelled_customer_ids))

        def _event(self, kind: str, **values: Any) -> None:
            if capture_events:
                self.events.append({"time": float(self.env.now), "event": kind, **values})

        def run_strict(self) -> None:
            # dvrpsim's public run() catches exceptions; let the research runner fail
            # loudly if an integration or simulation error occurs.
            self.init()
            self.on_simulation_start()
            self.env.run()
            self.on_simulation_finish()

        def routing_callback(self) -> dict[str, Any]:
            callback_started = time.perf_counter()
            state = self.get_state()
            try:
                time_now = float(state["time"])
                open_orders = state["open_orders"]
                expected_released = {
                    customer_id
                    for customer_id, release in release_times.items()
                    if release <= time_now + 1e-9
                }
                visible_ids = {
                    self.customer_by_order[order_id]
                    for order_id in open_orders
                    if order_id in self.customer_by_order
                }
                unexpected = sorted(visible_ids - expected_released)
                if unexpected:
                    self.leakage_violations.append(
                        {"time": time_now, "customer_ids": unexpected}
                    )
                    raise DynamicEpisodeError(
                        f"Unreleased customers leaked into routing state: {unexpected}"
                    )

                unassigned = {
                    self.customer_by_order[order_id]
                    for order_id, order_state in open_orders.items()
                    if order_state["assigned_vehicle"] is None
                    and order_id in self.customer_by_order
                }
                idle_at_depot: list[tuple[str, dict[str, Any]]] = []
                for vehicle_id, vehicle_state in state["vehicles"].items():
                    visit = vehicle_state["current_visit"]
                    if (
                        vehicle_state["status"] in {"IDLE", "WAITING_FOR_SERVICE"}
                        and visit is not None
                        and visit["location"] == "L000"
                        and visit["service_start_time"] is None
                    ) or (
                        vehicle_state["status"] == "IDLE"
                        and visit is not None
                        and visit["location"] == "L000"
                    ):
                        idle_at_depot.append((vehicle_id, vehicle_state))
                idle_at_depot.sort(key=lambda item: item[0])
                visible_instance, to_local = visible_problem(instance, self.released_customer_ids)
                to_original = {local: original for original, local in to_local.items()}

                decisions: dict[str, Any] = {"vehicles": {}, "orders": {}}
                for order_id in open_orders:
                    order = self.get_order_by_id(order_id)
                    if order is not None and order.is_without_decision:
                        decisions["orders"][order_id] = {"status": "accepted"}

                waiting = set(unassigned)
                for vehicle_id, _vehicle_state in idle_at_depot:
                    if not waiting:
                        break
                    routes_used = len(self.used_vehicle_ids | {vehicle_id})
                    route_ids = [to_original[local] for local in construct_route(
                        visible_instance, {to_local[cid] for cid in waiting}, time_now,
                        routes_used, policy_name, self.priority)]
                    if not route_ids:
                        continue
                    for customer_id in route_ids:
                        waiting.remove(customer_id)
                    order_ids = [self.order_by_customer[customer_id] for customer_id in route_ids]
                    visits = [
                        {"location": "L000", "pickup_list": order_ids},
                        *[
                            {
                                "location": f"L{customer_id:03d}",
                                "delivery_list": [self.order_by_customer[customer_id]],
                            }
                            for customer_id in route_ids
                        ],
                        {"location": "L000"},
                    ]
                    decisions["vehicles"][vehicle_id] = {"next_visits": visits}
                    self.used_vehicle_ids.add(vehicle_id)
                    self.route_records.append(
                        {
                            "vehicle_id": vehicle_id,
                            "assignment_time": time_now,
                            "dispatch_time": max(time_now, instance.depot.ready_time),
                            "customer_ids": list(route_ids),
                            "order_ids": order_ids,
                        }
                    )
                    self._event(
                        "dispatch",
                        vehicle_id=vehicle_id,
                        customer_ids=list(route_ids),
                        visible_customer_ids=sorted(visible_ids),
                    )

                self._event(
                    "routing_decision",
                    visible_customer_ids=sorted(visible_ids),
                    unassigned_customer_ids=sorted(unassigned),
                    idle_vehicle_ids=[vehicle_id for vehicle_id, _ in idle_at_depot],
                    assigned_vehicle_ids=sorted(decisions["vehicles"]),
                )
                return decisions
            finally:
                self.callback_latencies_ms.append(
                    (time.perf_counter() - callback_started) * 1000.0
                )

        def on_order_request(self, order: Order) -> None:
            self.released_customer_ids.add(self.customer_by_order[order.id])
            self._event(
                "order_release",
                order_id=order.id,
                customer_id=self.customer_by_order[order.id],
                release_date=float(order.release_date),
            )

        def on_order_rejection(self, order: Order) -> None:
            self.rejected_customer_ids.add(self.customer_by_order[order.id])
            self._event("order_rejected", order_id=order.id)

        def on_order_delivery(self, order: Order) -> None:
            customer_id = self.customer_by_order[order.id]
            self.delivered_customer_ids.add(customer_id)
            self._event("order_delivered", order_id=order.id, customer_id=customer_id)

        def on_order_cancellation(self, order: Order) -> None:
            self._event("order_cancelled", order_id=order.id,
                        customer_id=self.customer_by_order[order.id])
            super().on_order_cancellation(order)

        def on_vehicle_arrival(self, vehicle: Vehicle) -> None:
            self._event(
                "vehicle_arrival",
                vehicle_id=vehicle.id,
                location_id=vehicle.current_location.id,
            )
            if vehicle.current_location.id == "L000":
                for record in reversed(self.route_records):
                    if record["vehicle_id"] == vehicle.id and "return_time" not in record:
                        # The first depot visit picks up; only the final visit has no pickups.
                        if not vehicle.current_visit.pickup_list:
                            record["return_time"] = float(self.env.now)
                        break
                state = self.get_state()
                if any(
                    value["assigned_vehicle"] is None
                    for value in state["open_orders"].values()
                ):
                    self.request_for_routing()

        def on_vehicle_departure(self, vehicle: Vehicle) -> None:
            distance = float(vehicle.travel_distance(vehicle.previous_location, vehicle.next_location))
            self.total_distance += distance
            self.departure_count += 1
            self._event(
                "vehicle_departure",
                vehicle_id=vehicle.id,
                origin=vehicle.previous_location.id,
                destination=vehicle.next_location.id,
                distance=distance,
            )

        def on_vehicle_service_start(self, vehicle: Vehicle) -> None:
            super().on_vehicle_service_start(vehicle)
            visit = vehicle.current_visit
            for order in visit.delivery_list:
                customer_id = self.customer_by_order[order.id]
                self.service_starts[customer_id] = float(self.env.now)
                self._event(
                    "delivery_service_start",
                    vehicle_id=vehicle.id,
                    order_id=order.id,
                    customer_id=customer_id,
                    ready_time=instance.customers[customer_id].ready_time,
                    due_date=instance.customers[customer_id].due_date,
                )

        def on_simulation_finish(self) -> None:
            self.end_time = float(self.env.now)
            super().on_simulation_finish()

    model = SolomonModel()
    model.run_strict()

    violations = audit_routes(
        instance,
        model.route_records,
        model.service_starts,
        model.delivered_customer_ids,
        release_times=release_times,
        cancelled_customer_ids=model.cancelled_customer_ids,
    )
    for leakage in model.leakage_violations:
        violations.append({"type": "information_leakage", **leakage})
    if len(model.cancellation_records) != len(cancellation_requests):
        violations.append({"type": "missing_cancellation_event"})
    if model.cancelled_customer_ids & model.delivered_customer_ids:
        violations.append({"type": "cancelled_order_delivered"})
    for customer_id in model.cancelled_customer_ids:
        order = model.get_order_by_id(order_by_customer[customer_id])
        if order is not None and not order.is_canceled:
            violations.append({"type": "cancellation_state_mismatch", "customer_id": customer_id})
    for record in model.cancellation_records:
        if record["accepted"] and record["assigned_vehicle"] is not None:
            violations.append({"type": "assigned_order_cancelled", "customer_id": record["customer_id"]})
    if model.delivered_customer_ids != set(model.service_starts):
        violations.append(
            {
                "type": "delivery_service_start_coverage",
                "delivered": sorted(model.delivered_customer_ids),
                "service_starts": sorted(model.service_starts),
            }
        )

    requested = len(release_times)
    cancelled = len(model.cancelled_customer_ids)
    active = requested - cancelled
    served = len(model.delivered_customer_ids)
    on_time = sum(
        instance.customers[customer_id].ready_time - 1e-6
        <= model.service_starts.get(customer_id, math.inf)
        <= instance.customers[customer_id].due_date + 1e-6
        for customer_id in model.delivered_customer_ids
    )
    expected_distance = sum(
        instance.distance(origin, destination)
        for route in model.route_records
        for origin, destination in zip(
            [0, *route["customer_ids"]], [*route["customer_ids"], 0]
        )
    )
    if abs(expected_distance - model.total_distance) > 1e-6:
        violations.append(
            {
                "type": "distance_recalculation",
                "planned_distance": expected_distance,
                "simulator_distance": model.total_distance,
            }
        )

    latencies = model.callback_latencies_ms
    unserved_ids = sorted(set(release_times) - model.delivered_customer_ids - model.cancelled_customer_ids)
    reasons: dict[str, str] = {}
    for customer_id in unserved_ids:
        customer = instance.customers[customer_id]
        earliest = max(release_times[customer_id], instance.depot.ready_time) + instance.distance(0, customer_id)
        start = max(earliest, customer.ready_time)
        infeasible = (customer.demand > instance.capacity or start > customer.due_date + 1e-9 or
                      start + customer.service_time + instance.distance(customer_id, 0) > instance.depot.due_date + 1e-9)
        reasons[str(customer_id)] = ("rejected" if customer_id in model.rejected_customer_ids else
                                    "infeasible_at_release" if infeasible else "no_feasible_dispatch_before_horizon")
    cancellation_accepted = sum(record["accepted"] for record in model.cancellation_records)
    order_states = {str(cid): "COMPLETED" if cid in model.delivered_customer_ids else
                    "CANCELLED" if cid in model.cancelled_customer_ids else "UNSERVED"
                    for cid in release_times}
    model._event("episode_finished", order_states=order_states, unserved_reasons=reasons)
    result = {
        "episode_id": episode_id,
        "instance": instance.name.upper(),
        "policy": policy_name,
        "training_seed": policy.training_seed if policy else None,
        "expression": policy.expression if policy else (
            "EDD(due_date, customer_id)" if policy_name == "edd" else "GREEDY(distance, customer_id)"),
        "expression_sha256": policy.expression_sha256 if policy else None,
        "scenario_mode": scenario_mode,
        "scenario_seed": scenario_seed,
        "scenario_sha256": scenario.sha256,
        "scenario_rules": scenario.to_dict()["rules"],
        "cancellation_schedule": list(cancellation_requests),
        "dynamic_selected": sum(order.dynamic_selected for order in scenario.orders),
        "dynamic_released_after_zero": sum(order.release_time > 0 for order in scenario.orders),
        "cancel_candidates": len(scenario.cancellation_requests),
        "requested": requested,
        "released": len(model.released_customer_ids),
        "active_orders": active,
        "served": served,
        "on_time": int(on_time),
        "served_rate": served / active if active else 1.0,
        "served_rate_all_orders": served / requested if requested else 1.0,
        "on_time_rate": on_time / active if active else 1.0,
        "on_time_rate_served": on_time / served if served else 0.0,
        "late_served": served - int(on_time),
        "rejected": len(model.rejected_customer_ids),
        "unserved": len(unserved_ids),
        "unserved_customer_ids": unserved_ids,
        "unserved_reasons": reasons,
        "infeasible_orders": sum(reason == "infeasible_at_release" for reason in reasons.values()),
        "cancelled": cancelled,
        "cancelled_customer_ids": sorted(model.cancelled_customer_ids),
        "cancellation_requests": len(model.cancellation_records),
        "cancellation_accepted": cancellation_accepted,
        "cancellation_rejected": len(model.cancellation_records) - cancellation_accepted,
        "cancellation_success_rate": cancellation_accepted / len(model.cancellation_records) if model.cancellation_records else 0.0,
        "cancellation_records": model.cancellation_records,
        "order_states": order_states,
        "distance": model.total_distance,
        "travel_time": model.total_distance,
        "distance_per_served": model.total_distance / served if served else None,
        "expected_distance": expected_distance,
        "vehicles_used": len(model.used_vehicle_ids),
        "trip_count": len(model.route_records),
        "capacity_violations": sum(item["type"] == "capacity" for item in violations),
        "time_window_violations": sum(
            item["type"] in {"ready_time", "due_date", "depot_horizon"}
            for item in violations
        ),
        "information_leakage_violations": sum(
            item["type"] == "information_leakage" for item in violations
        ),
        "other_violations": sum(
            item["type"] not in {
                "capacity", "ready_time", "due_date", "depot_horizon", "information_leakage"
            }
            for item in violations
        ),
        "violation_count": len(violations),
        "violations_json": violations,
        "callback_count": len(latencies),
        "callback_latency_mean_ms": statistics.fmean(latencies) if latencies else 0.0,
        "callback_latency_p50_ms": _percentile(latencies, 50),
        "callback_latency_p95_ms": _percentile(latencies, 95),
        "callback_latency_max_ms": max(latencies, default=0.0),
        "simulation_time": model.end_time,
        "service_horizon": instance.depot.due_date,
        "wall_clock_seconds": time.perf_counter() - started,
        "release_times": {str(key): value for key, value in sorted(release_times.items())},
        "route_records": model.route_records,
        "served_customer_ids": sorted(model.delivered_customer_ids),
    }
    return result, model.events

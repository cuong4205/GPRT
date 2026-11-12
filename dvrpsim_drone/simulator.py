"""Dynamic benchmark episodes on DVRPSim's event and order lifecycle."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from dvrpsim import Model, Order, Vehicle

from dvrpsim_dynamic.simulator import _QuietLoggingCallback
from gprt_vrptw.evaluator import compile_tree
from gprt_vrptw.grammar import DRONE_FEATURES, Tree

from .data import DroneInstance, Settings
from .routing import EPS, Plan, Schedule, insertions, ready_departure, schedule

SIMULATOR_VERSION = "dvrpsim-drone-v2"


@dataclass(frozen=True)
class DispatchConfig:
    hold_minutes: float = 5.0
    gaps: int = 3
    drone_cost_ratio: float = 0.25
    drone_enabled: bool = True
    wait_at_depot: bool = True


@dataclass(frozen=True)
class Episode:
    routes: tuple[tuple[int, ...], ...]
    delivered: int
    truck_km: float
    drone_km: float
    drone_deliveries: int
    vehicles: int
    rendezvous_wait: float
    assignments_changed: int
    decision_seconds: tuple[float, ...] = field(compare=False)
    events: tuple[tuple, ...] = ()


class DroneModel(Model):
    def _init_logger(self):
        self.log = _QuietLoggingCallback(self)

    def __init__(self, instance, settings, config, tree, capture_events):
        super().__init__()
        self.instance = instance
        self.settings = settings
        self.config = config
        self.priority = compile_tree(tree, DRONE_FEATURES) if tree is not None else None
        self.capture_events = capture_events
        self.events = []
        self.decision_seconds = []
        self.pending = set()
        self.idle = set(range(instance.max_vehicles))
        self.vehicles_by_index = [Vehicle(f"V{i}") for i in range(instance.max_vehicles)]
        for vehicle in self.vehicles_by_index:
            self.add_vehicle(vehicle)
        self.plans = {}
        self.deadlines = {}
        self.wake_times = {}
        self.committed = set()
        self.completed = set()
        self.used = set()
        self.routes = []
        self.truck_km = self.drone_km = self.wait = 0.0
        self.drone_deliveries = self.changed = 0
        self.decision_at = None
        for cid in range(1, len(instance.customers)):
            order = Order(str(cid))
            order.release_date = instance.release[cid]
            order.quantity = instance.customers[cid].demand
            self.request_order(order)

    def record(self, kind, *details):
        if self.capture_events:
            self.events.append((float(self.env.now), kind, *details))

    def on_order_request(self, order):
        cid = int(order.id)
        self.pending.add(cid)
        self.record("release", cid)
        self.request_dispatch()

    def request_dispatch(self):
        if self.decision_at != self.env.now:
            self.decision_at = self.env.now
            self.env.process(self.dispatch())

    def rank(self, cid, vid, plan, previous, candidate, result, now):
        extra_truck = result.truck_km - previous.truck_km
        extra_drone = result.drone_km - previous.drone_km
        delta = extra_truck + self.config.drone_cost_ratio * extra_drone
        if self.priority is None:
            return delta
        customer = self.instance.customers[cid]
        delivered, _, mode = next(item for item in result.deliveries if item[1] == cid)
        horizon = self.instance.depot.due_date
        truck_capacity = self.settings.truck_capacity
        load = sum(self.instance.customers[item].demand for item in candidate.orders)
        return self.priority((
            (customer.due_date - delivered) / horizon,
            (result.finish - now) / horizon,
            (truck_capacity - load) / truck_capacity,
            customer.demand / truck_capacity,
            now / horizon,
            (now - self.instance.release[cid]) / horizon,
            extra_truck / (horizon * self.settings.truck_speed),
            extra_drone / (horizon * self.settings.drone_speed),
            result.rendezvous_wait / horizon,
            result.endurance_margin / self.settings.endurance,
            float(mode == "drone"),
            float(not plan.orders),
            (result.finish - previous.finish) / self.settings.truck_service,
            delta / (self.settings.truck_speed * self.settings.truck_service),
            result.service_wait / self.settings.truck_service,
            result.min_slack / self.settings.truck_service,
            (delivered - now) / self.settings.truck_service,
        ))

    def build(self, now):
        empty = Plan()
        empty_schedule = Schedule(now, (), 0, 0, 0, self.settings.endurance)
        plans = {vid: (empty, empty_schedule) for vid in sorted(self.idle)}
        # Deadline-first request order; the expression chooses vehicle, mode and insertion.
        for cid in sorted(self.pending, key=lambda i: (self.instance.customers[i].due_date, i)):
            if self.instance.customers[cid].due_date < now:
                continue
            best = None
            seen_empty = False
            for vid, (plan, prior) in plans.items():
                # Empty identical trucks give identical scores. Keep the first,
                # which also wins the vehicle-ID tie-break in the full search.
                if not plan.orders:
                    if seen_empty:
                        continue
                    seen_empty = True
                for index, candidate in enumerate(insertions(
                    plan, cid, self.config.drone_enabled and self.instance.drone_allowed[cid],
                    self.config.gaps,
                )):
                    result = schedule(self.instance, self.settings, candidate, now)
                    if result is None:
                        continue
                    key = (self.rank(cid, vid, plan, prior, candidate, result, now), vid, index)
                    if best is None or key < best[0]:
                        best = key, candidate, result
            if best is not None:
                key, candidate, result = best
                plans[key[1]] = candidate, result
        return {vid: plan for vid, (plan, _) in plans.items() if plan.orders}

    def dispatch(self):
        # Order releases and returns (priority 3) and hold timers (4) precede dispatch.
        yield self.env.timeout(0, priority=5)
        now = self.env.now
        if not self.idle or not self.pending or now >= self.instance.depot.due_date:
            return
        started = time.perf_counter()
        previous = {cid: vid for vid, plan in self.plans.items() for cid in plan.orders}
        plans = self.build(now)
        proposed = {cid: vid for vid, plan in plans.items() for cid in plan.orders}
        # Tentative assignments may return to the pending pool. They stay there
        # until a later feasible trip, while departed orders are excluded entirely.
        for cid in sorted(previous.keys() | proposed.keys()):
            old, new = previous.get(cid), proposed.get(cid)
            if old != new:
                self.changed += int(old is not None)
                self.record("assignment", cid, old, new)
        self.plans = plans
        for vid in tuple(self.deadlines):
            if vid not in plans:
                del self.deadlines[vid]
                self.wake_times.pop(vid, None)
        for vid, plan in sorted(plans.items()):
            due = self.deadlines.get(vid)
            if due is None:
                due = now + self.config.hold_minutes
                self.deadlines[vid] = due
            if self.config.wait_at_depot:
                due = max(due, ready_departure(self.instance, self.settings, plan, now))
            if now >= due or schedule(self.instance, self.settings, plan, due) is None:
                result = schedule(self.instance, self.settings, plan, now)
                if result is None:
                    raise RuntimeError("Infeasible departure")
                self.depart(vid, plan, result)
            elif self.wake_times.get(vid) != due:
                self.wake_times[vid] = due
                self.env.process(self.wake_at(vid, due))
        self.decision_seconds.append(time.perf_counter() - started)

    def wake_at(self, vid, due):
        yield self.env.timeout(due - self.env.now, priority=4)
        if self.wake_times.get(vid) == due:
            self.request_dispatch()

    def depart(self, vid, plan, result):
        ids = set(plan.orders)
        if len(ids) != len(plan.orders) or ids & self.committed or ids & self.completed or not ids <= self.pending:
            raise RuntimeError("Duplicate or unreleased order at departure")
        self.pending.difference_update(ids)
        self.committed.update(ids)
        self.idle.remove(vid)
        del self.plans[vid]
        del self.deadlines[vid]
        self.wake_times.pop(vid, None)
        self.used.add(vid)
        self.routes.append((vid, *plan.stops, -1, *plan.sorties))
        self.record("depart", vid, plan.stops, plan.sorties)
        for cid in plan.orders:
            order = self.get_order_by_id(str(cid))
            order.accept()
            order.pickup(self.vehicles_by_index[vid])
        self.env.process(self.execute(vid, result))

    def execute(self, vid, result):
        timeline = [(moment, "deliver", cid, mode) for moment, cid, mode in result.deliveries]
        timeline.extend((moment, event, cid, "") for moment, event, cid in result.sortie_events)
        airborne = None
        launched = 0.0
        for moment, event, cid, mode in sorted(timeline, key=lambda item: item[0]):
            yield self.env.medium_timeout(max(0, moment - self.env.now))
            if event == "launch":
                if airborne is not None or cid not in self.committed:
                    raise RuntimeError("Overlapping or unassigned sortie")
                airborne, launched = cid, self.env.now
                self.record(event, vid, cid)
            elif event == "recover":
                if airborne != cid or self.env.now - launched > self.settings.endurance + EPS:
                    raise RuntimeError("Invalid drone recovery")
                airborne = None
                self.record(event, vid, cid)
            else:
                customer = self.instance.customers[cid]
                if (cid not in self.committed
                        or not customer.ready_time <= self.env.now <= customer.due_date + EPS
                        or (mode == "drone" and airborne != cid)):
                    raise RuntimeError("Invalid service event")
                self.committed.remove(cid)
                self.completed.add(cid)
                self.get_order_by_id(str(cid)).deliver()
                self.drone_deliveries += int(mode == "drone")
                self.record("deliver", vid, cid, mode)
        if airborne is not None:
            raise RuntimeError("Vehicle returned before drone recovery")
        yield self.env.medium_timeout(max(0, result.finish - self.env.now))
        self.truck_km += result.truck_km
        self.drone_km += result.drone_km
        self.wait += result.rendezvous_wait
        self.idle.add(vid)
        self.record("return", vid)
        self.request_dispatch()


def run_episode(instance: DroneInstance, settings: Settings, config: DispatchConfig,
                tree: Tree | None, *, capture_events: bool = False) -> Episode:
    model = DroneModel(instance, settings, config, tree, capture_events)
    model.env.run()
    if model.committed or len(model.completed | model.pending) != instance.customer_count:
        raise RuntimeError("Episode did not account for all orders")
    return Episode(tuple(model.routes), len(model.completed), model.truck_km, model.drone_km,
                   model.drone_deliveries, len(model.used), model.wait, model.changed,
                   tuple(model.decision_seconds), tuple(model.events))

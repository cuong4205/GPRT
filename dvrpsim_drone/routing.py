"""Feasible depot plans. A sortie serves one customer across one truck leg."""
from __future__ import annotations

from dataclasses import dataclass

from .data import DroneInstance, Settings

EPS = 1e-8


@dataclass(frozen=True)
class Plan:
    stops: tuple[int, ...] = (0, 0)
    sorties: tuple[int, ...] = (0,)

    @property
    def orders(self) -> tuple[int, ...]:
        return self.stops[1:-1] + tuple(cid for cid in self.sorties if cid)


@dataclass(frozen=True)
class Schedule:
    finish: float
    deliveries: tuple[tuple[float, int, str], ...]
    truck_km: float
    drone_km: float
    rendezvous_wait: float
    endurance_margin: float
    sortie_events: tuple[tuple[float, str, int], ...] = ()
    service_wait: float = 0.0
    min_slack: float = 0.0


def schedule(instance: DroneInstance, settings: Settings, plan: Plan, start: float) -> Schedule | None:
    customers = instance.customers
    if sum(customers[cid].demand for cid in plan.orders) > settings.truck_capacity + EPS:
        return None
    if any(instance.release[cid] > start + EPS for cid in plan.orders):
        return None
    distance = instance.distance
    clock = start
    deliveries = []
    sortie_events = []
    truck_km = drone_km = wait = service_wait = 0.0
    min_slack = instance.depot.due_date - start
    margin = settings.endurance
    for origin, end, drone in zip(plan.stops, plan.stops[1:], plan.sorties):
        truck_leg = distance(origin, end)
        truck_km += truck_leg
        truck_arrival = clock + truck_leg / settings.truck_speed
        if drone:
            if not instance.drone_allowed[drone]:
                return None
            customer = customers[drone]
            first, second = distance(origin, drone), distance(drone, end)
            drone_km += first + second
            launch = clock + settings.launch
            arrival = launch + first / settings.drone_speed
            service_wait += max(0.0, customer.ready_time - arrival)
            delivered = max(arrival, customer.ready_time) + settings.drone_service
            if delivered > customer.due_date + EPS:
                return None
            drone_arrival = delivered + second / settings.drone_speed
            truck_arrival += settings.launch
            wait += abs(truck_arrival - drone_arrival)
            rendezvous = max(truck_arrival, drone_arrival) + settings.recovery
            margin = min(margin, settings.endurance - (rendezvous - clock))
            if margin < -EPS:
                return None
            sortie_events.extend(((clock, "launch", drone), (rendezvous, "recover", drone)))
            deliveries.append((delivered, drone, "drone"))
            min_slack = min(min_slack, customer.due_date - delivered)
            truck_arrival = rendezvous
        clock = truck_arrival
        if end:
            customer = customers[end]
            service_wait += max(0.0, customer.ready_time - clock)
            clock = max(clock, customer.ready_time) + settings.truck_service
            if clock > customer.due_date + EPS:
                return None
            deliveries.append((clock, end, "truck"))
            min_slack = min(min_slack, customer.due_date - clock)
    if clock > instance.depot.due_date + EPS:
        return None
    return Schedule(clock, tuple(sorted(deliveries)), truck_km, drone_km, wait, margin,
                    tuple(sortie_events), service_wait,
                    min(min_slack, instance.depot.due_date - clock))


def ready_departure(instance: DroneInstance, settings: Settings, plan: Plan, now: float) -> float:
    """Wait at the depot while every first-leg service would otherwise wait.

    Only already assigned customers are consulted. Delaying by the smaller wait
    lets the first useful service keep its earliest start. The full schedule is
    checked again before departure, including the drone endurance constraint.
    """
    end, drone = plan.stops[1], plan.sorties[0]
    waits = []
    if end:
        arrival = now + instance.distance(0, end) / settings.truck_speed
        if drone:
            arrival += settings.launch
        waits.append(max(0.0, instance.customers[end].ready_time - arrival))
    if drone:
        arrival = now + settings.launch + instance.distance(0, drone) / settings.drone_speed
        waits.append(max(0.0, instance.customers[drone].ready_time - arrival))
    return now + min(waits, default=0.0)


def insertions(plan: Plan, cid: int, drone_allowed: bool, limit: int):
    # Inspect the last available gaps; all policies see the same candidates.
    for index in range(len(plan.sorties) - 1, max(-1, len(plan.sorties) - limit - 1), -1):
        sortie = plan.sorties[index]
        stops = plan.stops[:index + 1] + (cid,) + plan.stops[index + 1:]
        before, after = plan.sorties[:index], plan.sorties[index + 1:]
        yield Plan(stops, before + (sortie, 0) + after)
        if sortie:
            yield Plan(stops, before + (0, sortie) + after)
        elif drone_allowed:
            yield Plan(plan.stops, before + (cid,) + after)

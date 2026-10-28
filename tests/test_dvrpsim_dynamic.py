from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from dvrpsim_dynamic.simulator import (
    audit_routes,
    make_release_times,
    parse_expression,
    run_episode,
    PolicySpec,
    load_selected_policies,
)
from dvrpsim_dynamic.scenario import Scenario, ScenarioConfig, ScenarioOrder, generate_scenario
from dvrpsim_dynamic.routing import visible_problem
from gprt_vrptw.evaluator import normalized_features
from vrptw_baselines.data import Customer, VRPTWInstance, parse_instance
from vrptw_baselines.solver import CandidateContext, solve
from validate_dvrpsim_results import validate_episode


def _tiny_instance() -> VRPTWInstance:
    customers = (
        Customer(0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0),
        Customer(1, 1.0, 0.0, 1.0, 0.0, 20.0, 1.0),
        Customer(2, 2.0, 0.0, 1.0, 0.0, 21.0, 1.0),
    )
    return VRPTWInstance(
        name="TINY",
        path=Path("tiny.txt"),
        max_vehicles=1,
        capacity=1.0,
        customers=customers,
        sha256=hashlib.sha256(b"tiny").hexdigest(),
        source="synthetic-test",
    )


def test_expression_reader_accepts_only_project_grammar() -> None:
    assert parse_expression("add(arrival_time, ready_time)") == (
        "add",
        "arrival_time",
        "ready_time",
    )


def test_dynamic_releases_are_seeded_and_individually_feasible() -> None:
    instance = _tiny_instance()
    first = make_release_times(instance, "dynamic", 42)
    second = make_release_times(instance, "dynamic", 42)
    assert first == second
    for customer_id, release in first.items():
        customer = instance.customers[customer_id]
        service_start = max(
            release + instance.distance(0, customer_id), customer.ready_time
        )
        assert service_start <= customer.due_date + 1e-9
        assert (
            service_start
            + customer.service_time
            + instance.distance(customer_id, 0)
            <= instance.depot.due_date + 1e-9
        )


def test_solomon_due_bound_means_service_start_not_service_completion() -> None:
    instance = parse_instance(Path("data/data/Solomon/c102.txt"))
    releases = make_release_times(instance, "dynamic", 20261025)
    customer = instance.customers[13]
    service_start = max(
        releases[13] + instance.distance(0, 13), customer.ready_time
    )
    assert customer.service_time == 90.0
    assert customer.due_date == 92.0
    assert releases[13] >= 0.0
    assert service_start <= customer.due_date + 1e-9


def test_independent_audit_detects_capacity_and_late_service() -> None:
    instance = _tiny_instance()
    violations = audit_routes(
        instance,
        [{"dispatch_time": 0.0, "customer_ids": [1, 2]}],
        {1: 1.0, 2: 30.0},
        {1, 2},
    )
    kinds = {item["type"] for item in violations}
    assert "capacity" in kinds
    assert "due_date" in kinds


def test_dvrpsim_releases_orders_and_does_not_route_hidden_customer() -> None:
    instance = _tiny_instance()
    result, events = run_episode(
        instance,
        {1: 0.0, 2: 5.0},
        "edd",
        scenario_mode="dynamic",
        scenario_seed=7,
        episode_id="tiny_dynamic",
    )
    first_decision = next(event for event in events if event["event"] == "routing_decision")
    assert first_decision["time"] == 0.0
    assert first_decision["visible_customer_ids"] == [1]
    assert result["served"] == 2
    assert result["unserved"] == 0
    assert result["violation_count"] == 0
    assert result["distance"] == result["expected_distance"]


def _run(instance=None, releases=None, cancellations=(), *, policy_name="edd", policy=None):
    instance = instance or _tiny_instance()
    releases = releases or {1: 0.0, 2: 5.0}
    return run_episode(instance, releases, policy_name, policy=policy, scenario_mode="dynamic",
                       scenario_seed=7, episode_id="test", cancellation_requests=cancellations)


def test_plan_scenario_has_exact_groups_and_preserves_benchmark(tmp_path):
    instance = parse_instance("data/data/Solomon/c101.txt")
    first = generate_scenario(instance, 42)
    assert first == generate_scenario(instance, 42)
    assert sum(o.dynamic_selected for o in first.orders) == 30
    assert len(first.cancellation_requests) == 5
    for order in first.orders:
        customer = instance.customers[order.customer_id]
        if order.dynamic_selected:
            assert 10 <= order.lead_time <= 15
            assert order.release_time == max(0, customer.ready_time - order.lead_time)
        else:
            assert order.release_time == 0
        if order.cancel_time is not None:
            assert 1 <= order.cancel_time - order.release_time <= 15
    path = tmp_path / "scenario.json"
    first.save(path)
    replay = Scenario.load(path, instance)
    assert replay == first
    assert replay.sha256 == first.sha256
    changed = generate_scenario(instance, 42, rules=ScenarioConfig(cancel_candidate_ratio=0.20))
    assert changed.release_times == first.release_times  # independent random streams
    assert parse_instance(instance.path).customers == instance.customers


def test_small_counts_use_half_up_and_selected_dynamic_can_release_at_zero():
    instance = _tiny_instance()
    scenario = generate_scenario(instance, rules=ScenarioConfig(dynamic_ratio=.25, cancel_candidate_ratio=.25))
    assert sum(o.dynamic_selected for o in scenario.orders) == 1  # 0.5 rounds up
    assert len(scenario.cancellation_requests) == 1
    assert all(o.release_time == 0 for o in scenario.orders)
    small = generate_scenario(instance)
    assert len(small.cancellation_requests) == 0
    static = generate_scenario(instance, mode="static")
    assert all(o.release_time == 0 and o.cancel_time is None for o in static.orders)


@pytest.mark.parametrize("cancel_time", [0.0, 2.0, 5.0])
def test_cancel_before_or_at_release_prevents_assignment(cancel_time):
    result, events = _run(cancellations=((2, cancel_time),))
    assert result["cancelled_customer_ids"] == [2]
    assert result["cancellation_accepted"] == 1
    assert result["served_customer_ids"] == [1]
    assert result["unserved"] == 0
    assert result["violation_count"] == 0
    assert all(2 not in e.get("customer_ids", []) for e in events if e["event"] == "dispatch")
    if cancel_time < 5:
        assert any(e["event"] == "order_release_suppressed" for e in events)
        assert not any(e["event"] == "order_release" and e["customer_id"] == 2 for e in events)


def test_cancel_same_time_as_initial_release_precedes_routing():
    result, events = _run(releases={1: 0.0, 2: 0.0}, cancellations=((1, 0), (2, 0)))
    assert result["served"] == 0
    assert result["cancelled"] == 2
    assert result["served_rate"] == 1
    kinds = [e["event"] for e in events]
    assert kinds.index("order_release") < kinds.index("cancellation_accepted") < kinds.index("routing_decision")


def test_cancel_assigned_order_before_departure_is_rejected():
    original = _tiny_instance()
    instance = replace(original, customers=(replace(original.depot, ready_time=10),
                                            *original.customers[1:]))
    result, events = _run(instance, {1: 0.0, 2: 50.0}, ((1, 1.0),))
    request = result["cancellation_records"][0]
    assert request["accepted"] is False and request["reason"] == "already_assigned"
    assert result["route_records"][0]["assignment_time"] == 0
    assert result["route_records"][0]["dispatch_time"] == 10
    assert result["served_customer_ids"] == [1]
    assert result["violation_count"] == 0
    assert any(e["event"] == "vehicle_departure" and e["time"] == 10 for e in events)


def test_cancel_completed_order_and_duplicate_request_are_rejected():
    result, _ = _run(cancellations=((1, 2.0), (2, 5.0), (2, 5.0)))
    assert [row["reason"] for row in result["cancellation_records"]] == [
        "already_completed", "unassigned", "already_cancelled"]
    assert result["cancellation_requests"] == 3
    assert result["cancellation_accepted"] == 1
    assert result["cancellation_rejected"] == 2
    assert result["cancellation_success_rate"] == pytest.approx(1 / 3)
    assert result["requested"] == result["served"] + result["cancelled"] + result["unserved"]


def test_busy_vehicle_returns_before_dispatching_new_order():
    result, events = _run(releases={1: 0.0, 2: 1.0})
    assert [r["assignment_time"] for r in result["route_records"]] == [0, 3]
    assert result["vehicles_used"] == 1 and result["trip_count"] == 2
    assert result["violation_count"] == 0
    assert result["travel_time"] == result["distance"] == 6


def test_waiting_unassigned_order_can_cancel_when_all_vehicles_busy():
    result, _ = _run(releases={1: 0.0, 2: 1.0}, cancellations=((2, 2.5),))
    assert result["cancellation_accepted"] == 1
    assert result["served"] == 1
    assert result["cancelled"] == 1


def test_release_return_and_cancel_at_same_time_are_batched():
    result, _ = _run(releases={1: 0.0, 2: 3.0}, cancellations=((2, 3.0),))
    assert result["served"] == 1 and result["cancelled"] == 1
    assert result["violation_count"] == 0


def test_no_feasible_vehicle_and_horizon_are_reported_without_dropping_requests():
    original = _tiny_instance()
    instance = replace(original, capacity=.5)
    result, events = _run(instance, cancellations=((2, 105.0),))
    assert result["served"] == 0 and result["cancelled"] == 1
    assert result["unserved_customer_ids"] == [1]
    assert result["unserved_reasons"] == {"1": "infeasible_at_release"}
    assert result["simulation_time"] == 105
    assert result["cancellation_requests"] == 1
    horizon = next(e for e in events if e["event"] == "horizon_reached")
    assert horizon["time"] == 100 and horizon["unserved_customer_ids"] == [1, 2]


def test_future_customer_data_cannot_change_visible_features_or_initial_dispatch():
    original = _tiny_instance()
    altered = replace(original, customers=(*original.customers[:2],
                         replace(original.customers[2], x=10000, due_date=10000)), sha256="altered")
    visible, mapping = visible_problem(original, {1})
    other, _ = visible_problem(altered, {1})
    context = CandidateContext(1, 0, 0, 1, 1, 1, 1, (1,), 1)
    assert mapping == {0: 0, 1: 1}
    assert normalized_features(visible, context) == normalized_features(other, context)
    left, _ = _run(original)
    right, _ = _run(altered)
    assert left["route_records"][0] == right["route_records"][0]


@pytest.mark.parametrize("method", ["gp", "gprt"])
def test_learned_expression_is_used_by_the_dynamic_router(method):
    instance = replace(_tiny_instance(), capacity=2)
    expression = "sub(0, d_current)"
    tree = parse_expression(expression)
    policy = PolicySpec(method, 1, expression, "hash", "test", "hash", None, tree)
    result, _ = _run(instance, {1: 0.0, 2: 0.0}, policy_name=method, policy=policy)
    assert result["route_records"][0]["customer_ids"] == [2, 1]  # farther first
    assert result["expression"] == expression
    assert result["violation_count"] == 0


def test_static_control_matches_the_shared_constructive_solver():
    instance = parse_instance("data/data/Solomon/c101.txt")
    scenario = generate_scenario(instance, mode="static")
    result, _ = run_episode(instance, scenario.release_times, "edd", scenario=scenario,
                            scenario_mode="static", scenario_seed=None, episode_id="static")
    expected = solve(instance, "edd")
    assert result["distance"] == pytest.approx(expected.distance)
    assert [r["customer_ids"] for r in result["route_records"]] == [list(r[1:-1]) for r in expected.routes]
    assert result["violation_count"] == 0


def test_scenario_replay_produces_same_decisions_and_event_log(tmp_path):
    instance = _tiny_instance()
    scenario = generate_scenario(instance, 42)
    path = tmp_path / "saved.json"
    scenario.save(path)
    outputs = [run_episode(instance, item.release_times, "edd", scenario=item,
                           scenario_mode="dynamic", scenario_seed=42, episode_id="repeat")
               for item in (scenario, Scenario.load(path, instance))]
    assert outputs[0][1] == outputs[1][1]
    for key in ("route_records", "cancellation_records", "order_states", "distance", "scenario_sha256"):
        assert outputs[0][0][key] == outputs[1][0][key]


def test_policy_loader_honors_roots_and_requested_seeds(tmp_path):
    directory = tmp_path / "policies/gp/123"
    directory.mkdir(parents=True)
    (directory / "final_heuristic.txt").write_text("d_current\n", encoding="utf-8")
    policies = load_selected_policies(tmp_path, ("gp",), policy_root="policies", training_seeds=(123,))
    assert len(policies) == 1 and policies[0].training_seed == 123
    with pytest.raises(RuntimeError, match="Missing validation-selected"):
        load_selected_policies(tmp_path, ("gp",), policy_root="policies", training_seeds=(999,))


def test_invalid_scenario_and_instance_data_are_rejected(tmp_path):
    instance = _tiny_instance()
    with pytest.raises(ValueError, match="capacity"):
        generate_scenario(replace(instance, capacity=-1))
    with pytest.raises(ValueError, match="time window"):
        generate_scenario(replace(instance, customers=(instance.depot,
            replace(instance.customers[1], ready_time=30), instance.customers[2])))
    with pytest.raises(ValueError, match="each customer"):
        _run(releases={1: 0.0})
    with pytest.raises(ValueError, match="finite"):
        _run(releases={1: float("nan"), 2: 0})
    scenario = generate_scenario(instance)
    path = tmp_path / "bad.json"
    payload = scenario.to_dict()
    payload["instance_sha256"] = "different"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        Scenario.load(path, instance)


def test_malformed_solomon_rows_are_rejected(tmp_path):
    raw = Path("data/data/Solomon/c101.txt").read_text()
    lines = raw.splitlines()
    row_index = next(i for i, line in enumerate(lines) if line.split()[:1] == ["100"])
    lines[row_index] = "100 1 2 3 4 5"  # missing service time
    path = tmp_path / "malformed.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed customer row"):
        parse_instance(path)


def test_saved_metrics_can_be_reconciled_with_events_and_corruption_is_detected():
    instance = _tiny_instance()
    scenario = generate_scenario(instance, 7)
    result, events = run_episode(instance, scenario.release_times, "edd", scenario=scenario,
                                scenario_mode="dynamic", scenario_seed=7, episode_id="audit")
    assert validate_episode(result, events, scenario, instance) == []
    damaged = {**result, "served": result["served"] + 1}
    assert "served differs from event log" in validate_episode(damaged, events, scenario, instance)


def test_upstream_tutorial_request_order_release_dates():
    # Official tutorial_02: five requests released at 8, 16, 24, 32, 40.
    from dvrpsim import Location, Model, Order
    from dvrpsim_dynamic.simulator import _QuietLoggingCallback

    class TutorialModel(Model):
        def _init_logger(self):
            self.log = _QuietLoggingCallback(self)

        def on_order_request(self, order):
            releases.append((order.id, self.env.now))

    releases = []
    model = TutorialModel()
    depot = Location("DEPOT")
    model.add_location(depot)
    for index in range(5):
        location = Location(f"CUSTOMER {index + 1}")
        model.add_location(location)
        order = Order(f"O-{index + 1}")
        order.pickup_location, order.delivery_location = depot, location
        order.release_date = (index + 1) * 8
        model.request_order(order)
    model.run()
    assert releases == [(f"O-{index + 1}", (index + 1) * 8) for index in range(5)]

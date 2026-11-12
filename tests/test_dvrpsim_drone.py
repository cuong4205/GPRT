from pathlib import Path

from dvrpsim_drone.data import Settings, load_instance
from dvrpsim_drone.evaluator import DynamicEvaluator
from dvrpsim_drone.simulator import DispatchConfig, run_episode
from gprt_vrptw.grammar import DRONE_FEATURES, Grammar, prefix_to_tree, tree_to_prefix
from gprt_vrptw.splits import family, stratified_sample


SETTINGS = Settings(1, 2, 1, 1, 1, 1, 5, 30, 1, 1)


def instance(tmp_path: Path, rows: str, fleet: int = 2):
    path = tmp_path / "small.csv"
    path.write_text(
        "x,y,demand,open,close,servicetime,drone_serve,time\n"
        "0,0,0,0,100,5,1,0\n" + rows, encoding="utf-8"
    )
    return load_instance(path, SETTINGS, fleet)


def test_tentative_assignment_moves_and_departed_trip_is_locked(tmp_path):
    data = instance(tmp_path, "4,0,1,0,40,5,0,0\n5,0,1,5,20,5,0,2\n2,0,1,6,60,5,0,6\n")
    result = run_episode(data, SETTINGS, DispatchConfig(), 0.0, capture_events=True)
    assert (0.0, "assignment", 1, None, 0) in result.events
    assert (2.0, "assignment", 1, 0, 1) in result.events
    assert (5.0, "depart", 0, (0, 2, 0), (0, 0)) in result.events
    assert (6.0, "release", 3) in result.events
    assert not any(event[1] == "assignment" and event[2] == 2 and event[0] >= 5
                   for event in result.events)
    assert (11.0, "deliver", 0, 2, "truck") in result.events
    assert result.delivered == 3


def test_release_at_departure_is_considered_first(tmp_path):
    data = instance(tmp_path, "4,0,1,0,40,5,0,0\n5,0,1,5,20,5,0,5\n")
    result = run_episode(data, SETTINGS, DispatchConfig(), 0.0, capture_events=True)
    assert result.events.index((5.0, "release", 2)) < result.events.index(
        (5.0, "depart", 0, (0, 2, 0), (0, 0))
    )


def test_drone_timeline_and_reloaded_policy_match_training(tmp_path):
    data = instance(tmp_path, "4,0,1,0,40,5,1,0\n", fleet=1)
    dispatch = DispatchConfig()
    tree = ("sub", 0.0, "is_drone")
    tokens = tree_to_prefix(tree)
    assert Grammar(features=DRONE_FEATURES).validate_expression(tokens)
    evaluator = DynamicEvaluator(SETTINGS, dispatch, 0.0)
    trained = evaluator.solution(tree, data)
    loaded = prefix_to_tree(tokens)
    assert trained == run_episode(data, SETTINGS, dispatch, loaded)
    trace = run_episode(data, SETTINGS, dispatch, loaded, capture_events=True)
    assert (5.0, "launch", 0, 1) in trace.events
    assert (9.0, "deliver", 0, 1, "drone") in trace.events
    assert (12.0, "recover", 0, 1) in trace.events
    assert (12.0, "return", 0) in trace.events


def test_waiting_at_depot_keeps_vehicle_available_for_new_urgent_order(tmp_path):
    data = instance(tmp_path, "4,0,1,40,60,5,0,0\n2,0,1,7,15,5,0,7\n", fleet=1)
    legacy = run_episode(data, SETTINGS, DispatchConfig(wait_at_depot=False),
                         "trip_duration", capture_events=True)
    revised = run_episode(data, SETTINGS, DispatchConfig(), "trip_duration", capture_events=True)
    assert legacy.delivered == 1
    assert revised.delivered == 2
    assert (5.0, "depart", 0, (0, 1, 0), (0, 0)) in legacy.events
    assert (7.0, "depart", 0, (0, 2, 0), (0, 0)) in revised.events
    assert not any(event[1] == "depart" and event[0] < 7 for event in revised.events)


def test_future_unreleased_order_does_not_change_departure_decisions(tmp_path):
    data = instance(tmp_path, "4,0,1,40,60,5,0,0\n2,0,1,7,15,5,0,7\n", fleet=1)
    first = run_episode(data, SETTINGS, DispatchConfig(), "extra_duration", capture_events=True)
    changed = instance(tmp_path, "4,0,1,40,60,5,0,0\n80,0,1,70,95,5,0,7\n", fleet=1)
    second = run_episode(changed, SETTINGS, DispatchConfig(), "extra_duration", capture_events=True)
    assert [e for e in first.events if e[0] < 7] == [e for e in second.events if e[0] < 7]


def test_dynamic_instance_names_preserve_time_window_families():
    from types import SimpleNamespace
    assert family("h100r101") == "R1"
    assert family("h100r201") == "R2"
    assert family("h400RC2_4_1") == "RC2"
    assert family("c101") == "C1"
    items = [SimpleNamespace(name=f"h100{kind}{n:02}")
             for kind in ("c2", "r1", "r2") for n in range(1, 5)]
    selected = stratified_sample(items, 6)
    assert {family(item.name) for item in selected} == {"C2", "R1", "R2"}

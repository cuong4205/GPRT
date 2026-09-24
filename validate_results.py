from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from run_experiment import summarize
from vrptw_baselines.data import discover_instances
from vrptw_baselines.solver import validate_solution


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate saved VRPTW experiment output")
    parser.add_argument("--results", type=Path, default=Path("outputs/vrptw_baselines/results.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/data/Solomon"))
    args = parser.parse_args()

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    instances = {instance.name.upper(): instance for instance in discover_instances(args.data_dir)}
    assert len(instances) == 56
    assert payload["metadata"]["solomon_instance_count"] == 56
    assert payload["metadata"]["test_count"] == 12
    assert len(payload["details"]) == 56 * 5 * 5
    assert len(payload["run_summary"]) == 5 * 5
    assert sum(row["split"] == "test" for row in payload["details"]) == 12 * 5 * 5

    grouped_routes: dict[tuple[int, int, str, str], list[tuple[int, ...]]] = defaultdict(list)
    for row in payload["routes"]:
        key = (row["run"], row["seed"], row["method"], row["instance"])
        grouped_routes[key].append(tuple(int(value) for value in row["route"].split("-")))

    for detail in payload["details"]:
        key = (detail["run"], detail["seed"], detail["method"], detail["instance"])
        routes = grouped_routes[key]
        checked = validate_solution(instances[detail["instance"]], routes)
        assert checked["feasible"] == detail["feasible"], key
        assert len(routes) == detail["vehicle_count"], key
        assert math.isclose(float(checked["distance"]), detail["distance"], abs_tol=1e-7), key
        assert math.isclose(float(checked["waiting_time"]), detail["waiting_time"], abs_tol=1e-7), key

    run_summary, method_summary = summarize(payload["details"])
    assert run_summary == payload["run_summary"]
    assert method_summary == payload["method_summary"]
    assert all(row["expression"] == "due" for row in payload["gp_runs"])
    print(
        f"PASS: {len(payload['details'])} solutions, {len(payload['routes'])} routes, "
        f"{len(payload['method_summary'])} methods"
    )


if __name__ == "__main__":
    main()

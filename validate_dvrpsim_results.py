"""Reconcile saved dynamic episode metrics with scenario files and event logs."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from dvrpsim_dynamic.scenario import Scenario
from dvrpsim_dynamic.simulator import audit_routes, sha256_file
from vrptw_baselines.data import VRPTWInstance, discover_instances


def validate_episode(result: dict[str, Any], events: list[dict[str, Any]],
                     scenario: Scenario, instance: VRPTWInstance) -> list[str]:
    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    counts = Counter(event["event"] for event in events)
    check(result["scenario_sha256"] == scenario.sha256, "scenario hash mismatch")
    check(result["requested"] == instance.customer_count, "requested count mismatch")
    check(result["served"] + result["cancelled"] + result["unserved"] == result["requested"], "order accounting mismatch")
    for metric, kind in (("served", "order_delivered"), ("released", "order_release"),
                         ("cancellation_requests", "cancellation_request"),
                         ("cancellation_accepted", "cancellation_accepted"),
                         ("cancellation_rejected", "cancellation_rejected")):
        check(result[metric] == counts[kind], f"{metric} differs from event log")
    check(result["cancellation_requests"] == len(scenario.cancellation_requests), "cancellation schedule count mismatch")
    check(result["cancellation_requests"] == result["cancellation_accepted"] + result["cancellation_rejected"], "cancellation accounting mismatch")
    check(result["cancelled"] == result["cancellation_accepted"], "duplicate successful cancellation")
    check(all(a["time"] <= b["time"] for a, b in zip(events, events[1:])), "events not in time order")

    released: set[int] = set()
    assigned: set[int] = set()
    cancelled: set[int] = set()
    delivered: set[int] = set()
    starts: dict[int, float] = {}
    cancellations = []
    for event in events:
        kind, now = event["event"], event["time"]
        cid = event.get("customer_id")
        if kind == "order_release":
            check(cid not in released and cid not in cancelled, "duplicate or cancelled release")
            check(now == scenario.release_times[cid], "wrong release time")
            released.add(cid)
        elif kind == "dispatch":
            ids = set(event["customer_ids"])
            check(ids <= released and not ids & (assigned | cancelled), "hidden/duplicate/cancelled assignment")
            assigned.update(ids)
        elif kind == "routing_decision":
            check(set(event["visible_customer_ids"]) <= released - cancelled, "hidden/cancelled routing state")
        elif kind == "cancellation_request":
            cancellations.append((cid, now))
        elif kind == "cancellation_accepted":
            check(cid not in assigned | cancelled | delivered, "assigned/completed/duplicate cancellation")
            cancelled.add(cid)
        elif kind == "delivery_service_start":
            check(cid in assigned and cid not in cancelled, "invalid service start")
            starts[cid] = now
        elif kind == "order_delivered":
            check(cid in starts and cid not in cancelled | delivered, "invalid delivery")
            delivered.add(cid)
    check(sorted(cancellations) == sorted(scenario.cancellation_requests), "cancellation times differ from scenario")
    check(sorted(delivered) == result["served_customer_ids"], "served IDs differ from events")
    check(sorted(cancelled) == result["cancelled_customer_ids"], "cancelled IDs differ from events")
    unserved = set(scenario.release_times) - delivered - cancelled
    check(sorted(unserved) == result["unserved_customer_ids"], "unserved IDs differ from events")
    check(set(result["unserved_reasons"]) == {str(cid) for cid in unserved}, "unserved reasons incomplete")
    on_time = sum(instance.customers[cid].ready_time - 1e-6 <= starts.get(cid, math.inf) <= instance.customers[cid].due_date + 1e-6
                  for cid in delivered)
    check(on_time == result["on_time"], "on-time metric differs from events")
    active = instance.customer_count - len(cancelled)
    check(math.isclose(result["served_rate"], len(delivered) / active if active else 1), "served rate denominator mismatch")
    distance = sum(e["distance"] for e in events if e["event"] == "vehicle_departure")
    check(math.isclose(result["distance"], distance, abs_tol=1e-6), "distance differs from events")
    check(math.isclose(result["travel_time"], distance, abs_tol=1e-6), "travel time mismatch")
    check(not audit_routes(instance, result["route_records"], starts, delivered,
                           release_times=scenario.release_times, cancelled_customer_ids=cancelled), "independent route audit failed")
    check(result["violation_count"] == 0, "episode reports constraint violations")
    return errors


def validate_directory(directory: Path) -> dict[str, Any]:
    run = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parent
    instances = {instance.name.lower(): instance for instance in discover_instances(root / run["effective_config"]["solomon_root"])}

    def keyed(filename: str) -> dict[str, Any]:
        rows = [json.loads(line) for line in (directory / filename).read_text(encoding="utf-8").splitlines() if line]
        lookup = {row["episode_id"]: row for row in rows}
        if len(lookup) != len(rows):
            raise ValueError(f"Duplicate episode IDs in {filename}")
        return lookup

    results, logs, manifests = (keyed(name) for name in ("episode_results.jsonl", "events.jsonl", "episode_manifest.jsonl"))
    if not results or results.keys() != logs.keys() or results.keys() != manifests.keys():
        raise ValueError("Episode result/event/manifest coverage mismatch")
    failures = {}
    for episode_id, result in results.items():
        instance = instances[result["instance"].lower()]
        manifest = manifests[episode_id]
        scenario = Scenario.load(manifest["scenario_file"], instance)
        errors = validate_episode(result, logs[episode_id]["events"], scenario, instance)
        if manifest["scenario_sha256"] != scenario.sha256:
            errors.append("episode manifest scenario hash mismatch")
        if errors:
            failures[episode_id] = errors
    return {"passed": not failures, "episodes": len(results), "failures": failures,
            "source_files_changed_since_run": [path for path, digest in run["source_sha256"].items()
                                               if sha256_file(root / path) != digest]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    report = validate_directory(args.results_dir)
    (args.results_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

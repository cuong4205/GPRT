from __future__ import annotations

import pytest

from scripts.report_dvrpsim_performance import pairs, pm, summarize


def row(method="gp", seed=1, served_ids=(1,), callbacks=1, latency=1.0):
    served = len(served_ids)
    return {"policy": method, "training_seed": seed, "instance": "TINY", "scenario_seed": 7,
            "scenario_sha256": "same", "served": served, "unserved": 2 - served, "cancelled": 0,
            "on_time_rate": served / 2, "distance": 4., "distance_per_served": 4. / served,
            "vehicles_used": 1, "trip_count": 1, "wall_clock_seconds": .1,
            "callback_latency_p95_ms": latency, "callback_count": callbacks,
            "callback_latency_mean_ms": latency, "cancellation_requests": 0,
            "cancellation_accepted": 0, "cancellation_rejected": 0, "violation_count": 0,
            "unserved_reasons": {"2": "infeasible_at_release"}, "cancellation_records": [],
            "served_customer_ids": list(served_ids), "cancelled_customer_ids": []}


def test_report_callback_mean_is_weighted_and_single_policy_has_no_seed_sd():
    summary = summarize([row(callbacks=2, latency=1), row(callbacks=1, latency=10)])
    assert summary["callback_weighted_mean_ms"] == 4
    assert summary["routing_wall_seconds_per_episode"] == .006
    assert "±" not in pm(summary, "distance")


def test_report_pairs_check_actual_workload_and_materialized_scenario():
    gp, gprt = row(), row("gprt", served_ids=(2,))
    assert pairs([gp, gprt])[0]["same_completed_and_cancelled_ids"] is False
    gprt["served_customer_ids"] = [1]
    assert pairs([gp, gprt])[0]["same_completed_and_cancelled_ids"] is True
    gprt["scenario_sha256"] = "different"
    with pytest.raises(ValueError, match="different scenarios"):
        pairs([gp, gprt])

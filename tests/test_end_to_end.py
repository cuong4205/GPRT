from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pytest

import run_end_to_end as pipeline
from gprt_vrptw.config import GPRTConfig
from run_gprt_experiment import run_gp_baseline
from vrptw_baselines.data import Customer, VRPTWInstance
from vrptw_baselines.solver import solve


def args_for(out: Path) -> argparse.Namespace:
    return argparse.Namespace(output_dir=out, training_config=pipeline.ROOT / "configs/gprt_main.json",
                              dynamic_config=pipeline.ROOT / "configs/dvrpsim_dynamic.json",
                              data_dir=None, device="cpu", route_backend="python", threads=1)


def tiny_instance() -> VRPTWInstance:
    return VRPTWInstance(name="TINY", path=Path("tiny.txt"), max_vehicles=2, capacity=2,
                        customers=(Customer(0, 0, 0, 0, 0, 100, 0), Customer(1, 1, 0, 1, 0, 20, 1),
                                   Customer(2, 2, 0, 1, 0, 20, 1)),
                        sha256=hashlib.sha256(b"tiny-e2e").hexdigest(), source="synthetic-test")


def test_plan_is_32_cycles_with_all_validation_instances_and_new_artifacts(tmp_path):
    plan = pipeline.build_plan(args_for(tmp_path / "new-run"))
    training, dynamic = plan["training"], plan["dynamic"]
    assert training["cycles"] == 32 and training["skip_test"]
    assert set(training["methods"]) == {"gp", "gprt"}
    assert training["require_full_train_coverage"]
    assert dynamic["policy_root"] == training["output_dir"]
    assert "reduced_main" not in json.dumps(dynamic)
    assert "policy_roots" not in dynamic
    validation = {k for k, v in plan["split"].items() if v == "validation"}
    test = {k for k, v in plan["split"].items() if v == "test"}
    assert len(validation) == len(test) == 12
    assert set(dynamic["pilot_validation_instances"]) == validation
    assert validation.isdisjoint(test)
    assert plan["expected_episodes"] == {"dynamic_validation": 396, "dynamic_test": 528, "static_test": 132}
    assert not (tmp_path / "new-run").exists()


def test_gp_records_actual_completed_cycles_and_coverage(monkeypatch):
    monkeypatch.setenv("GPRT_ROUTE_BACKEND", "python")
    first = tiny_instance()
    config = GPRTConfig(cycles=32, population_size=4, transformer_samples=2, gp_generations=1,
                        max_depth=2, max_tokens=7, screening_train_size=1, validation_candidates=2)
    _, _, search = run_gp_baseline([first], [first], config, 42)
    assert search["completed_cycles"] == 32
    assert search["training_coverage_rate"] == 1.0
    assert search["seen_training_instances"] == ["tiny"]


def test_gp_runner_writes_32_cycle_metadata_used_by_pipeline(tmp_path, monkeypatch):
    from dataclasses import replace
    import run_gprt_experiment as runner
    from validate_gprt_results import validate
    from vrptw_baselines.data import discover_instances
    # Keep the canonical 56 names/split, with tiny synthetic coordinates for a bounded CLI test.
    names = discover_instances(pipeline.ROOT / "data/data/Solomon")
    fixtures = [replace(tiny_instance(), name=i.name, source="Solomon", sha256=i.name) for i in names]
    monkeypatch.setattr(runner, "discover_instances", lambda _: fixtures)
    config = tmp_path / "config.json"
    output = tmp_path / "run"
    pipeline.write_json(config, {"methods": ["gp"], "seeds": [42], "cycles": 32,
                        "screening_run": True, "skip_test": True, "population_size": 4,
                        "transformer_samples": 2, "gp_generations": 1, "max_tokens": 7,
                        "max_depth": 2, "screening_train_size": 2, "validation_candidates": 2,
                        "output_dir": str(output), "route_backend": "python"})
    monkeypatch.setattr("sys.argv", ["run_gprt_experiment.py", "--config", str(config)])
    runner.main()
    summary = pipeline.read_json(output / "gp/42/training_summary.json")
    assert summary["completed_cycles"] == summary["training_coverage_count"] == 32
    assert (output / "gp/42/final_heuristic.txt").read_text(encoding="utf-8").strip() == summary["best_expression"]
    assert validate(output)["test_skipped"]


def test_gprt_completes_32_cycles_and_saves_selected_text(tmp_path, monkeypatch):
    from dataclasses import replace
    from gprt_vrptw.trainer import GPRTTrainer
    monkeypatch.setenv("GPRT_ROUTE_BACKEND", "python")
    base = tiny_instance()
    train = [replace(base, name=f"T{i:02}", sha256=f"train-{i}") for i in range(32)]
    validation = [replace(base, name="V01", sha256="validation-only")]
    split = {i.name.lower(): "train" for i in train}
    split["v01"] = "validation"
    config = GPRTConfig(cycles=32, d_model=16, nhead=2, num_layers=1, dim_feedforward=32,
                        population_size=4, transformer_samples=2, gp_generations=1,
                        max_depth=2, max_tokens=7, screening_train_size=2,
                        validation_candidates=2, elite_screening_candidates=2)
    GPRTTrainer(config).train(train, validation, seed=42, output_dir=tmp_path, split_manifest=split)
    summary = pipeline.read_json(tmp_path / "training_summary.json")
    assert summary["completed_cycles"] == summary["gradient_steps"] == 32
    assert summary["training_coverage_count"] == 32
    assert summary["training_coverage_rate"] == 1.0
    assert (tmp_path / "final_heuristic.txt").read_text(encoding="utf-8").strip() == summary["best_expression"]
    assert (tmp_path / "best.pt").is_file()


def make_training_artifacts(out, plan):
    training = out / "training"
    split = plan["split"]
    config = {"cycles": 32}
    pipeline.write_json(training / "results.json", {"protocol": {"config": config,
                        "methods": list(pipeline.METHODS), "seeds": pipeline.SEEDS}})
    pipeline.write_json(training / "split_manifest.json", split)
    for method in pipeline.METHODS:
        for seed in pipeline.SEEDS:
            directory = training / method / str(seed)
            pipeline.write_json(directory / "training_summary.json", {
                "completed_cycles": 32, "training_coverage_count": 32, "training_coverage_rate": 1.0,
                "seen_training_instances": sorted(k for k, v in split.items() if v == "train"),
                "best_expression": "due_date"})
            pipeline.write_json(directory / "config.json", config)
            pipeline.write_json(directory / "split_manifest.json", split)
            (directory / "final_heuristic.txt").write_text("due_date\n", encoding="utf-8")


def test_lock_rejects_incomplete_training_and_detects_changed_heuristic(tmp_path, monkeypatch):
    import validate_gprt_results
    monkeypatch.setattr(validate_gprt_results, "validate", lambda _: {"test_skipped": True})
    plan = pipeline.build_plan(args_for(tmp_path))
    make_training_artifacts(tmp_path, plan)
    summary_path = tmp_path / "training/gp" / str(pipeline.SEEDS[0]) / "training_summary.json"
    summary = pipeline.read_json(summary_path)
    summary["completed_cycles"] = 31
    pipeline.write_json(summary_path, summary)
    with pytest.raises(RuntimeError, match="Incomplete"):
        pipeline.lock_heuristics(tmp_path, plan)
    assert not (tmp_path / "heuristics_manifest.json").exists()
    summary["completed_cycles"] = 32
    pipeline.write_json(summary_path, summary)
    pipeline.lock_heuristics(tmp_path, plan)
    records = pipeline.verify_heuristics(tmp_path)
    assert len(records) == 10
    Path(records[0]["file"]).write_text("arrival_time\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Frozen heuristic changed"):
        pipeline.verify_heuristics(tmp_path)


def test_static_evaluation_uses_saved_expressions_and_one_edd_policy(tmp_path, monkeypatch):
    from vrptw_baselines import data
    instance = tiny_instance()
    monkeypatch.setattr(data, "discover_instances", lambda _: [instance])
    records = []
    for method in pipeline.METHODS:
        file = tmp_path / f"{method}.txt"
        file.write_text("due_date\n", encoding="utf-8")
        records.append({"method": method, "seed": 42, "file": str(file), "sha256": pipeline.digest(file), "expression": "due_date"})
    pipeline.write_json(tmp_path / "heuristics_manifest.json", {"policies": records})
    plan = {"training": {"data_dir": ".", "route_backend": "python"}, "split": {"tiny": "test"},
            "dataset": {"tiny": instance.sha256}, "expected_episodes": {"static_test": 3}}
    pipeline.run_static_test(tmp_path, plan)
    rows = pipeline.read_json(tmp_path / "static_test/results.json")["rows"]
    assert len(rows) == 3
    assert {r["method"] for r in rows} == {"gp", "gprt", "edd"}
    assert next(r for r in rows if r["method"] == "edd")["seed"] is None
    assert all(r["served"] == 2 and r["feasible"] for r in rows)
    assert pipeline.read_json(tmp_path / "static_test/gate_report.json")["passed"]


def test_static_audit_preserves_unserved_as_quality_outcome():
    from dataclasses import replace
    instance = tiny_instance()
    impossible = replace(instance, max_vehicles=0)
    solution = solve(impossible, "edd")
    assert solution.unserved
    assert pipeline.audit_static_solution(impossible, solution) == []


def test_dynamic_bridge_reads_text_on_all_12_validation_instances(tmp_path):
    # Real simulator/validator integration using two fixed test expressions.
    # These are fixtures, not claimed 32-cycle training results.
    plan = pipeline.build_plan(args_for(tmp_path))
    seed = pipeline.SEEDS[0]
    plan["dynamic"].update(training_seeds=[seed], pilot_scenario_seeds=[20261025])
    plan["expected_episodes"]["dynamic_validation"] = 12 * 2 * 3
    pipeline.write_json(tmp_path / "dynamic_config.json", plan["dynamic"])
    pipeline.write_json(tmp_path / "training/split_manifest.json", plan["split"])
    records = []
    for method, expression in (("gp", "due_date"), ("gprt", "add(arrival_time, ready_time)")):
        file = tmp_path / "training" / method / str(seed) / "final_heuristic.txt"
        file.parent.mkdir(parents=True)
        file.write_text(expression + "\n", encoding="utf-8")
        records.append({"method": method, "seed": seed, "file": str(file),
                        "sha256": pipeline.digest(file), "expression": expression})
    pipeline.write_json(tmp_path / "heuristics_manifest.json", {"policies": records})
    pipeline.run_dynamic_phase(tmp_path, plan, "pilot", os.environ.copy())
    manifest = pipeline.read_json(tmp_path / "dynamic/pilot/run_manifest.json")
    assert len(manifest["instances"]) == 12
    assert {i["split"] for i in manifest["instances"]} == {"validation"}
    assert pipeline.read_json(tmp_path / "dynamic/pilot/validation_report.json")["passed"]
    assert pipeline.read_json(tmp_path / "dynamic/pilot/gate_report.json")["episodes"] == 72


def test_validation_failure_blocks_all_test_and_resume_skips_completed_train(tmp_path, monkeypatch):
    plan = pipeline.build_plan(args_for(tmp_path))
    calls = []

    def result(name):
        path = tmp_path / f"{name}.json"
        pipeline.write_json(path, {"stage": name})
        return [path]

    def train(command, log, env):
        calls.append("train")
        assert "--skip-test" in command
        assert env["OMP_NUM_THREADS"] == "1"
        pipeline.write_json(tmp_path / "training/results.json", {"done": True})

    def lock(*_):
        calls.append("lock")
        return result("locked")

    def dynamic(out, plan, phase, env):
        calls.append(phase)
        if phase == "pilot":
            raise RuntimeError("validation gate failed")
        return result(phase)

    monkeypatch.setattr(pipeline, "run_command", train)
    monkeypatch.setattr(pipeline, "lock_heuristics", lock)
    monkeypatch.setattr(pipeline, "run_dynamic_phase", dynamic)
    monkeypatch.setattr(pipeline, "run_static_test", lambda *_: (calls.append("static_test") or result("static")))
    monkeypatch.setattr(pipeline, "write_report", lambda *_: (calls.append("report") or result("report")))
    with pytest.raises(RuntimeError, match="validation gate"):
        pipeline.run_pipeline(tmp_path, plan)
    assert calls == ["train", "lock", "pilot"]
    assert pipeline.read_json(tmp_path / "pipeline_manifest.json")["status"] == "failed"
    monkeypatch.setattr(pipeline, "run_dynamic_phase", lambda out, plan, phase, env: (calls.append(phase) or result(phase)))
    pipeline.run_pipeline(tmp_path, plan, resume=True)
    assert calls == ["train", "lock", "pilot", "pilot", "static_test", "final", "report"]
    assert pipeline.read_json(tmp_path / "pipeline_manifest.json")["status"] == "complete"
    changed = pipeline.read_json(tmp_path / "dynamic_config.json")
    changed["scenario"]["dynamic_ratio"] = 0.1
    pipeline.write_json(tmp_path / "dynamic_config.json", changed)
    with pytest.raises(ValueError, match="Frozen config changed"):
        pipeline.run_pipeline(tmp_path, plan, resume=True)


def test_pipeline_refuses_existing_unowned_output(tmp_path):
    (tmp_path / "keep.txt").write_text("existing user data", encoding="utf-8")
    plan = pipeline.build_plan(args_for(tmp_path))
    with pytest.raises(FileExistsError):
        pipeline.run_pipeline(tmp_path, plan)
    assert (tmp_path / "keep.txt").read_text(encoding="utf-8") == "existing user data"


def test_report_keeps_training_cost_separate_from_online_results(tmp_path):
    pipeline.write_json(tmp_path / "heuristics_manifest.json", {"policies": []})
    pipeline.write_json(tmp_path / "static_test/results.json", {"rows": [
        {"method": "gp", "feasible": True, "served": 100, "unserved": 0, "vehicles": 10, "distance": 1000, "runtime_ms": 5}]})
    final = tmp_path / "dynamic/final/episode_results.jsonl"
    final.parent.mkdir(parents=True)
    final.write_text(json.dumps({"scenario_mode": "dynamic", "policy": "gp", "served": 90,
                     "unserved": 9, "cancelled": 1, "on_time_rate": 90/99, "distance": 2000,
                     "distance_per_served": 2000/90, "callback_latency_p95_ms": 3}) + "\n", encoding="utf-8")
    pipeline.write_json(tmp_path / "training/results.json", {"searches": [
        {"method": "gp", "end_to_end_seconds": 120, "instance_evaluations": 1000, "peak_rss_mb": 256}]})
    pipeline.write_report(tmp_path)
    summary = pipeline.read_json(tmp_path / "pipeline_summary.json")
    assert summary["cycles"] == 32
    assert summary["offline_training"][0]["end_to_end_seconds"] == 120
    assert summary["simulator_test"][0]["served"] == 90
    assert summary["simulator_test"][0]["callback_latency_p95_ms"] == 3
    assert (tmp_path / "end_to_end_report.md").is_file()

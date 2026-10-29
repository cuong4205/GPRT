"""Train GP/GPRT for 32 cycles, freeze heuristics, validate, then test static/dynamic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
METHODS = ("gp", "gprt")
SEEDS = list(range(20260914, 20260919))
STAGES = ("train", "lock_heuristics", "dynamic_validation", "static_test", "dynamic_test", "report")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A stopped run must leave either the previous manifest or a complete new one.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    from gprt_vrptw.config import GPRTConfig
    from gprt_vrptw.splits import split_solomon
    from dvrpsim_dynamic.scenario import ScenarioConfig
    from vrptw_baselines.data import discover_instances

    out = args.output_dir.resolve()
    training = read_json(args.training_config)
    dynamic = read_json(args.dynamic_config)
    data = (args.data_dir or ROOT / training.get("data_dir", "data/data/Solomon")).resolve()
    instances = discover_instances(data)
    if len(instances) != 56 or any(i.source != "Solomon" for i in instances):
        raise ValueError("Expected exactly 56 Solomon instances")
    split = split_solomon(instances)
    if training.get("seeds", SEEDS) != SEEDS:
        raise ValueError(f"This main protocol uses the five locked training seeds {SEEDS}")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    training.update(methods=list(METHODS), cycles=32, seeds=SEEDS, skip_test=True,
                    screening_run=False, require_full_train_coverage=True,
                    output_dir=str(out / "training"), data_dir=str(data),
                    device=args.device, route_backend=args.route_backend)
    config = GPRTConfig(**{k: v for k, v in training.items() if k in GPRTConfig.__dataclass_fields__})
    config.validate()
    # Every validation instance is used for simulator validation, never train/test instances.
    dynamic.update(methods=["gprt", "gp", "edd"], training_seeds=SEEDS,
                   policy_root=str(out / "training"), solomon_root=str(data),
                   split_manifest=str(out / "training" / "split_manifest.json"),
                   pilot_validation_instances=sorted(k for k, v in split.items() if v == "validation"))
    dynamic.pop("policy_roots", None)  # Never inherit per-method paths to legacy heuristics.
    ScenarioConfig(**dynamic["scenario"])
    for field in ("pilot_scenario_seeds", "test_scenario_seeds"):
        values = dynamic[field]
        if not values or len(values) != len(set(values)) or any(type(v) is not int for v in values):
            raise ValueError(f"{field} must contain unique integer seeds")
    versions = {name: version(name) for name in ("torch", "numpy", "dvrpsim", "simpy")}
    if versions["dvrpsim"] != dynamic["dvrpsim_version"]:
        raise ValueError(f"Expected dvrpsim=={dynamic['dvrpsim_version']}")
    sources = [Path(__file__), ROOT / "run_gprt_experiment.py", ROOT / "run_dvrpsim_dynamic.py",
               ROOT / "validate_gprt_results.py", ROOT / "validate_dvrpsim_results.py"]
    for directory in ("gprt_vrptw", "dvrpsim_dynamic", "vrptw_baselines"):
        sources.extend(sorted((ROOT / directory).glob("*.py")))
    return {
        "training": training, "dynamic": dynamic, "split": split,
        "dataset": {i.name.lower(): i.sha256 for i in instances},
        "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
        "versions": versions, "python": sys.version, "threads": args.threads,
        "stages": list(STAGES),
        "expected_episodes": {
            "dynamic_validation": 12 * (1 + len(dynamic["pilot_scenario_seeds"])) * 11,
            "dynamic_test": 12 * (1 + len(dynamic["test_scenario_seeds"])) * 11,
            "static_test": 12 * 11,
        },
    }


def run_command(command: list[str], log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            if process.wait():
                raise RuntimeError(f"Stage failed; see {log}")
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise


def lock_heuristics(out: Path, plan: dict[str, Any]) -> list[Path]:
    from dvrpsim_dynamic.simulator import load_selected_policies
    from validate_gprt_results import validate

    training = out / "training"
    report = validate(training)
    payload = read_json(training / "results.json")
    protocol = payload["protocol"]
    if (protocol["config"]["cycles"] != 32 or set(protocol["methods"]) != set(METHODS)
            or protocol["seeds"] != SEEDS or not report["test_skipped"]
            or read_json(training / "split_manifest.json") != plan["split"]):
        raise RuntimeError("Training results do not match the locked 32-cycle protocol")
    policies = load_selected_policies(ROOT, METHODS, policy_root=training, training_seeds=tuple(SEEDS))
    records, files = [], [training / "results.json", training / "split_manifest.json"]
    expected_train = sorted(k for k, v in plan["split"].items() if v == "train")
    for policy in policies:
        directory = training / policy.name / str(policy.training_seed)
        summary = read_json(directory / "training_summary.json")
        if (summary["completed_cycles"] != 32 or summary["training_coverage_count"] != 32
                or summary["training_coverage_rate"] != 1.0
                or sorted(summary["seen_training_instances"]) != expected_train
                or summary["best_expression"] != policy.expression):
            raise RuntimeError(f"Incomplete or mismatched training: {directory}")
        if (read_json(directory / "config.json") != protocol["config"]
                or read_json(directory / "split_manifest.json") != plan["split"]):
            raise RuntimeError(f"Training config/split mismatch: {directory}")
        records.append({"method": policy.name, "seed": policy.training_seed,
                        "file": policy.source_file, "sha256": policy.source_sha256,
                        "expression": policy.expression, "completed_cycles": 32})
        files.extend(directory / name for name in ("final_heuristic.txt", "training_summary.json", "config.json", "split_manifest.json"))
    write_json(out / "heuristics_manifest.json", {"selection": "static validation only; frozen before all test evaluation", "policies": records})
    write_json(training / "validation_report.json", report)
    return [out / "heuristics_manifest.json", training / "validation_report.json", *files]


def verify_heuristics(out: Path) -> list[dict[str, Any]]:
    records = read_json(out / "heuristics_manifest.json")["policies"]
    for item in records:
        if digest(Path(item["file"])) != item["sha256"]:
            raise RuntimeError(f"Frozen heuristic changed: {item['file']}")
    return records


def run_dynamic_phase(out: Path, plan: dict[str, Any], phase: str, env: dict[str, str]) -> list[Path]:
    from validate_dvrpsim_results import validate_directory

    locked = verify_heuristics(out)
    run_command([sys.executable, "-u", str(ROOT / "run_dvrpsim_dynamic.py"), "--config",
                 str(out / "dynamic_config.json"), "--phase", phase, "--output-dir", str(out / "dynamic")],
                out / "logs" / f"dynamic_{phase}.log", env)
    directory = out / "dynamic" / phase
    manifest = read_json(directory / "run_manifest.json")
    actual = {(p["policy"], p["training_seed"]): p["source_sha256"] for p in manifest["policies"]}
    expected = {(p["method"], p["seed"]): p["sha256"] for p in locked}
    split_name = "validation" if phase == "pilot" else "test"
    names = {k for k, v in plan["split"].items() if v == split_name}
    if (actual != expected or {i["name"].lower() for i in manifest["instances"]} != names
            or any(i["sha256"] != plan["dataset"][i["name"].lower()] for i in manifest["instances"])):
        raise RuntimeError("Simulator did not use the frozen heuristics or the full intended split")
    gate = read_json(directory / "gate_report.json")
    count_key = "dynamic_validation" if phase == "pilot" else "dynamic_test"
    report = validate_directory(directory)
    write_json(directory / "validation_report.json", report)
    if (not gate["all_episodes_constraint_clean"] or not report["passed"]
            or report["source_files_changed_since_run"] or gate["episodes"] != plan["expected_episodes"][count_key]):
        raise RuntimeError(f"Simulator {phase} failed independent validation")
    verify_heuristics(out)
    return [directory / name for name in ("run_manifest.json", "gate_report.json", "validation_report.json",
            "episode_results.jsonl", "episode_manifest.jsonl", "events.jsonl")]


def audit_static_solution(instance: Any, solution: Any) -> list[str]:
    from vrptw_baselines.solver import validate_solution

    checked = validate_solution(instance, solution.routes)
    errors = []
    if (checked["duplicates"] or any(not r.feasible for r in checked["route_metrics"])
            or len(solution.routes) > instance.max_vehicles):
        errors.append("route constraint violation")
    if (sorted(solution.unserved) != checked["missing"] or solution.feasible != checked["feasible"]
            or not math.isclose(solution.distance, checked["distance"], abs_tol=1e-6)):
        errors.append("solution/checker mismatch")
    # Missing customers are a quality outcome, not a reason to suppress an episode.
    return errors


def run_static_test(out: Path, plan: dict[str, Any]) -> list[Path]:
    from dvrpsim_dynamic.simulator import parse_expression
    from gprt_vrptw.evaluator import FitnessEvaluator, distance_scale
    from gprt_vrptw.splits import family
    from vrptw_baselines.data import discover_instances
    from vrptw_baselines.solver import solve

    policies = verify_heuristics(out)
    policies = [*policies, {"method": "edd", "seed": None, "expression": "EDD(due_date, customer_id)", "sha256": None}]
    instances = [i for i in discover_instances(Path(plan["training"]["data_dir"])) if plan["split"][i.name.lower()] == "test"]
    if any(i.sha256 != plan["dataset"][i.name.lower()] for i in instances):
        raise RuntimeError("Static test data changed since the protocol was locked")
    directory = out / "static_test"
    directory.mkdir(exist_ok=True)
    rows, errors = [], []
    for policy in policies:
        evaluator = FitnessEvaluator(route_backend=plan["training"]["route_backend"])
        tree = parse_expression(policy["expression"]) if policy["method"] != "edd" else None
        for instance in sorted(instances, key=lambda i: i.name.lower()):
            started = time.perf_counter()
            solution = evaluator.solution(tree, instance) if tree is not None else solve(instance, "edd")
            elapsed = (time.perf_counter() - started) * 1000
            violations = audit_static_solution(instance, solution)
            if violations:
                errors.append({"method": policy["method"], "seed": policy["seed"], "instance": instance.name, "errors": violations})
            rows.append({"method": policy["method"], "seed": policy["seed"], "instance": instance.name.upper(),
                         "family": family(instance.name), "feasible": solution.feasible,
                         "served": instance.customer_count - len(solution.unserved), "unserved": len(solution.unserved),
                         "vehicles": solution.vehicle_count, "distance": solution.distance,
                         "normalized_distance": solution.distance / (instance.customer_count * distance_scale(instance)),
                         "runtime_ms": elapsed, "expression": policy["expression"], "heuristic_sha256": policy["sha256"],
                         "routes": [list(r) for r in solution.routes]})
    write_json(directory / "results.json", {"split": "test", "heuristics_manifest_sha256": digest(out / "heuristics_manifest.json"),
               "fleet_semantics": "one route per physical vehicle; original Solomon fleet bound", "rows": rows})
    _csv(directory / "test_results.csv", rows)
    write_json(directory / "gate_report.json", {"passed": not errors, "rows": len(rows), "errors": errors})
    if errors or len(rows) != plan["expected_episodes"]["static_test"]:
        raise RuntimeError("Static test failed independent route verification")
    verify_heuristics(out)
    return [directory / name for name in ("results.json", "test_results.csv", "gate_report.json")]


def summarize(rows: list[dict[str, Any]], group_keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    return [{**dict(zip(group_keys, key)), "observations": len(items),
             **{metric: statistics.fmean(float(r[metric]) for r in items) for metric in metrics}}
            for key, items in sorted(groups.items(), key=lambda p: str(p[0]))]


def write_report(out: Path) -> list[Path]:
    verify_heuristics(out)
    static = summarize(read_json(out / "static_test" / "results.json")["rows"], ("method",),
                       ("feasible", "served", "unserved", "vehicles", "distance", "runtime_ms"))
    rows = [json.loads(line) for line in (out / "dynamic" / "final" / "episode_results.jsonl").read_text(encoding="utf-8").splitlines() if line]
    dynamic = summarize(rows, ("scenario_mode", "policy"), ("served", "unserved", "cancelled", "on_time_rate",
                        "distance", "distance_per_served", "callback_latency_p95_ms"))
    searches = read_json(out / "training" / "results.json")["searches"]
    costs = summarize(searches, ("method",), ("end_to_end_seconds", "instance_evaluations", "peak_rss_mb"))
    write_json(out / "pipeline_summary.json", {"cycles": 32, "static_test": static, "simulator_test": dynamic, "offline_training": costs})
    lines = ["# Kết quả end-to-end GP/GPRT 32 cycle", "", "Biểu thức chọn trên validation tĩnh, khóa trước validation động và mọi test.",
             "", "## Test tĩnh (benchmark, một tuyến/xe)", "", "| Policy | Số lần | Khả thi (%) | Phục vụ | Xe | Quãng đường |", "|---|---:|---:|---:|---:|---:|"]
    for r in static:
        lines.append(f"| {r['method']} | {r['observations']} | {100*r['feasible']:.2f} | {r['served']:.2f} | {r['vehicles']:.2f} | {r['distance']:.2f} |")
    lines += ["", "## Test trong simulator (xe có thể chạy nhiều chuyến)", "", "| Điều kiện | Policy | Số lần | Phục vụ | Chưa phục vụ | Hủy | Quãng đường | TB p95 (ms) |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in dynamic:
        lines.append(f"| {r['scenario_mode']} | {r['policy']} | {r['observations']} | {r['served']:.2f} | {r['unserved']:.2f} | {r['cancelled']:.2f} | {r['distance']:.2f} | {r['callback_latency_p95_ms']:.3f} |")
    lines += ["", "Các số là trung bình theo lần mô phỏng; EDD là một policy, không phải năm model độc lập.",
              "Đọc mức phục vụ/hủy cùng quãng đường; dùng paired_comparisons.csv để đối chiếu cùng instance và scenario.",
              "Không suy ra ưu thế thống kê từ trung bình. Chi phí học offline lưu riêng trong pipeline_summary.json.",
              "Static control trong simulator khác benchmark tĩnh về quyền tái sử dụng xe; không so trực tiếp với BKS single-trip.",
              "Kịch bản là tổng hợp từ Solomon; các test instance này đã xuất hiện trong pilot lịch sử nên không phải một bộ test xác nhận mới hoàn toàn."]
    (out / "end_to_end_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [out / "pipeline_summary.json", out / "end_to_end_report.md"]


def run_pipeline(out: Path, plan: dict[str, Any], *, resume: bool = False) -> None:
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    manifest_path = out / "pipeline_manifest.json"
    if resume:
        manifest = read_json(manifest_path)
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Config, source, environment or data changed; use a new output directory")
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Output directory is not empty; use a new directory or --resume")
        out.mkdir(parents=True, exist_ok=True)
        manifest = {"fingerprint": fingerprint, "plan": plan, "stages": {}, "status": "running"}
        write_json(out / "training_config.json", plan["training"])
        write_json(out / "dynamic_config.json", plan["dynamic"])
        write_json(manifest_path, manifest)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS=str(plan["threads"]), MKL_NUM_THREADS=str(plan["threads"]),
               PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    actions = {
        "train": lambda: (run_command([sys.executable, "-u", str(ROOT / "run_gprt_experiment.py"), "--config",
                                      str(out / "training_config.json"), "--skip-test"], out / "logs" / "train.log", env)
                          or [out / "training" / "results.json"]),
        "lock_heuristics": lambda: lock_heuristics(out, plan),
        "dynamic_validation": lambda: run_dynamic_phase(out, plan, "pilot", env),
        "static_test": lambda: run_static_test(out, plan),
        "dynamic_test": lambda: run_dynamic_phase(out, plan, "final", env),
        "report": lambda: write_report(out),
    }
    for stage in STAGES:
        for filename, expected_config in (("training_config.json", plan["training"]), ("dynamic_config.json", plan["dynamic"])):
            if read_json(out / filename) != expected_config:
                raise ValueError(f"Frozen config changed: {filename}")
        saved = manifest["stages"].get(stage, {})
        if saved.get("status") == "complete":
            for name, expected in saved["artifacts"].items():
                if digest(out / name) != expected:
                    raise RuntimeError(f"Completed stage artifact changed: {name}")
            print(f"[skip] {stage}", flush=True)
            continue
        print(f"[start] {stage}", flush=True)
        state = {"status": "running", "started_at_utc": datetime.now(timezone.utc).isoformat()}
        manifest["stages"][stage] = state
        write_json(manifest_path, manifest)
        try:
            files = actions[stage]()
            state.update(status="complete", artifacts={str(p.relative_to(out)): digest(p) for p in files},
                         completed_at_utc=datetime.now(timezone.utc).isoformat())
            write_json(manifest_path, manifest)
        except BaseException as error:
            state.update(status="failed", error=str(error))
            manifest["status"] = "failed"
            write_json(manifest_path, manifest)
            raise
    manifest["status"] = "complete"
    write_json(manifest_path, manifest)
    print(f"Completed: {out / 'end_to_end_report.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-config", type=Path, default=ROOT / "configs/gprt_main.json")
    parser.add_argument("--dynamic-config", type=Path, default=ROOT / "configs/dvrpsim_dynamic.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/end_to_end_32cycle")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--route-backend", choices=("auto", "python", "numba"), default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Skip stages completed with identical inputs; incomplete training restarts")
    parser.add_argument("--dry-run", action="store_true", help="Print effective configuration and stages without writing or training")
    args = parser.parse_args()
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    run_pipeline(args.output_dir.resolve(), plan, resume=args.resume)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from importlib.metadata import version

from dvrpsim_dynamic.simulator import (
    PolicySpec,
    load_selected_policies,
    run_episode,
    sha256_file,
)
from dvrpsim_dynamic.scenario import Scenario, ScenarioConfig, generate_scenario
from gprt_vrptw.splits import family, split_solomon
from vrptw_baselines.data import discover_instances


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "dvrpsim_dynamic.json"
OUTPUT_ROOT = ROOT / "outputs" / "dvrpsim_dynamic_30pct_cancel5"
METRICS = (
    "served_rate",
    "on_time_rate",
    "unserved",
    "distance",
    "distance_per_served",
    "vehicles_used",
    "served",
    "on_time",
    "cancelled",
    "cancellation_requests",
    "cancellation_accepted",
    "cancellation_rejected",
    "cancellation_success_rate",
    "late_served",
    "travel_time",
    "wall_clock_seconds",
    "callback_latency_p95_ms",
)


def _json_line(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _csv_row(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.dumps(value, ensure_ascii=False, sort_keys=True)
        if isinstance(value, (dict, list))
        else value
        for key, value in result.items()
        if key not in {"route_records", "served_customer_ids", "cancellation_records", "order_states"}
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], row["training_seed"], row["scenario_mode"])].append(row)
    output = []
    for (policy, seed, mode), items in sorted(
        grouped.items(), key=lambda item: (item[0][0], str(item[0][1]), item[0][2])
    ):
        record: dict[str, Any] = {
            "policy": policy,
            "training_seed": seed,
            "scenario_mode": mode,
            "episodes": len(items),
            "instances": len({item["instance"] for item in items}),
            "scenario_seeds": len({item["scenario_seed"] for item in items}),
        }
        for metric in METRICS:
            values = [float(item[metric]) for item in items if item[metric] is not None]
            record[f"{metric}_mean"] = statistics.fmean(values) if values else None
            record[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        record["capacity_violations"] = sum(int(item["capacity_violations"]) for item in items)
        record["time_window_violations"] = sum(
            int(item["time_window_violations"]) for item in items
        )
        record["information_leakage_violations"] = sum(
            int(item["information_leakage_violations"]) for item in items
        )
        output.append(record)
    return output


def _paired_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        lookup[
            (
                row["instance"],
                row["scenario_mode"],
                row["scenario_seed"],
                row["policy"],
                row["training_seed"],
            )
        ] = row

    comparisons: list[dict[str, Any]] = []
    combinations = sorted(
        {(row["scenario_mode"], row["training_seed"]) for row in rows if row["policy"] == "gprt"},
        key=lambda value: (value[0], str(value[1])),
    )
    for mode, seed in combinations:
        for baseline in ("gp", "edd", "greedy"):
            deltas: dict[str, list[float]] = defaultdict(list)
            for gprt_row in rows:
                if (
                    gprt_row["policy"] != "gprt"
                    or gprt_row["training_seed"] != seed
                    or gprt_row["scenario_mode"] != mode
                ):
                    continue
                baseline_seed = seed if baseline == "gp" else None
                key = (
                    gprt_row["instance"],
                    mode,
                    gprt_row["scenario_seed"],
                    baseline,
                    baseline_seed,
                )
                baseline_row = lookup.get(key)
                if baseline_row is None:
                    continue
                for metric in METRICS:
                    left, right = gprt_row[metric], baseline_row[metric]
                    if left is not None and right is not None:
                        deltas[metric].append(float(left) - float(right))
            for metric, values in deltas.items():
                comparisons.append(
                    {
                        "scenario_mode": mode,
                        "training_seed": seed,
                        "comparison": f"gprt_minus_{baseline}",
                        "metric": metric,
                        "paired_episodes": len(values),
                        "mean_delta": statistics.fmean(values) if values else None,
                        "sd_delta": statistics.stdev(values) if len(values) > 1 else 0.0,
                    }
                )
    return comparisons


def _policy_manifest(policy: PolicySpec) -> dict[str, Any]:
    return {
        "policy": policy.name,
        "training_seed": policy.training_seed,
        "expression": policy.expression,
        "expression_sha256": policy.expression_sha256,
        "source_file": policy.source_file,
        "source_sha256": policy.source_sha256,
        "checkpoint_sha256": policy.checkpoint_sha256,
        "selection": "final_heuristic.txt saved by the training run after validation selection",
    }


def _episode_id(
    instance: str, policy_name: str, policy_seed: int | None, mode: str, scenario_seed: int | None
) -> str:
    return "_".join(
        (
            instance.lower(),
            policy_name,
            str(policy_seed) if policy_seed is not None else "none",
            mode,
            str(scenario_seed) if scenario_seed is not None else "fixed",
        )
    )


def _run_phase(config: dict[str, Any], phase: str, output_dir: Path, *,
               config_path: Path = CONFIG_PATH, scenario_file: Path | None = None) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = output_dir / "episodes.csv"
    manifest_path = output_dir / "episode_manifest.jsonl"
    events_path = output_dir / "events.jsonl"
    results_path = output_dir / "episode_results.jsonl"
    for path in (episodes_path, manifest_path, events_path, results_path):
        if path.exists():
            path.unlink()

    all_instances = discover_instances(ROOT / config["solomon_root"])
    split = split_solomon(all_instances)
    saved_split = json.loads((ROOT / config["split_manifest"]).read_text(encoding="utf-8"))
    if saved_split != split:
        raise RuntimeError("Saved split manifest does not match the locked Solomon split")
    if len(all_instances) != 56:
        raise RuntimeError(f"Expected 56 Solomon instances, found {len(all_instances)}")
    instances_by_name = {instance.name.lower(): instance for instance in all_instances}
    methods = tuple(config["methods"])
    if not methods or len(set(methods)) != len(methods) or set(methods) - {"gp", "gprt", "edd", "greedy"}:
        raise ValueError("methods must be a nonempty unique selection of gprt, gp, edd, greedy")
    seed_filter = tuple(config["training_seeds"]) if config.get("training_seeds") else None
    policies = [policy for method in methods if method in {"gp", "gprt"}
                for policy in load_selected_policies(ROOT, (method,),
                    policy_root=config.get("policy_roots", {}).get(method, config.get("policy_root", "outputs/gprt_vrptw/reduced_main")),
                    training_seeds=seed_filter)]
    policies_by_method: dict[str, list[PolicySpec]] = {
        name: [policy for policy in policies if policy.name == name]
        for name in ("gprt", "gp")
    }
    if phase in {"pilot", "smoke"}:
        names = [name.lower() for name in config["pilot_validation_instances"]]
        scenario_seeds = [int(seed) for seed in config["pilot_scenario_seeds"]]
        expected_split = "validation"
    else:
        names = sorted(name for name, part in split.items() if part == "test")
        scenario_seeds = [int(seed) for seed in config["test_scenario_seeds"]]
        expected_split = "test"
    if any(split.get(name) != expected_split for name in names):
        raise RuntimeError(f"{phase} instances do not match the locked {expected_split} split")
    if phase == "final" and len(names) != 12:
        raise RuntimeError(f"Expected 12 held-out Solomon test instances, got {len(names)}")
    replay = None
    if phase == "smoke":
        names, scenario_seeds = names[:1], scenario_seeds[:1]
        policies_by_method = {name: values[:1] for name, values in policies_by_method.items()}
        if scenario_file is not None:
            raw_scenario = json.loads(scenario_file.read_text(encoding="utf-8"))
            name = str(raw_scenario["instance"]).lower()
            if name not in instances_by_name:
                raise ValueError(f"Unknown replay instance: {name}")
            names = [name]
            expected_split = split[name]
            replay = Scenario.load(scenario_file, instances_by_name[name])
    policies = [policy for values in policies_by_method.values() for policy in values]
    rules = ScenarioConfig(**config["scenario"])
    # Record local code identity without modifying Git configuration or index.
    git_args = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
    revision = subprocess.run([*git_args, "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=False)
    dirty = subprocess.run([*git_args, "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True, check=False)

    manifest = {
        "phase": phase,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "simulator": "dvrpsim",
        "simulator_version": version("dvrpsim"),
        "pinned_version": config["dvrpsim_version"],
        "split_manifest_sha256": sha256_file(ROOT / config["split_manifest"]),
        "config_sha256": sha256_file(config_path),
        "effective_config": config,
        "algorithm_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "working_tree_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path)
                          for path in [Path(__file__), *sorted((ROOT / "dvrpsim_dynamic").glob("*.py")),
                                       ROOT / "gprt_vrptw/evaluator.py", ROOT / "gprt_vrptw/grammar.py",
                                       ROOT / "vrptw_baselines/data.py", ROOT / "vrptw_baselines/solver.py"]},
        "simpy_version": version("simpy"),
        "distance": config["distance"],
        "dispatch_policy": config["dispatch_policy"],
        "scenario_rules": config["scenario"],
        "same_timestamp_order": "native vehicle/release (priority 3), cancellation (4), routing (5), horizon (6)",
        "normalization": "released customer data only; original grammar and feature formulas",
        "fleet_semantics": "fixed fleet, reusable vehicles; capacity resets at each depot trip",
        "rate_denominator": "requested minus successfully cancelled; all-order rate also saved",
        "release_seed_set": scenario_seeds,
        "instances": [
            {
                "name": instances_by_name[name].name.upper(),
                "split": split[name],
                "sha256": instances_by_name[name].sha256,
            }
            for name in names
        ],
        "policies": [_policy_manifest(policy) for policy in policies],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    result_rows: list[dict[str, Any]] = []
    episode_count = 0
    for name in names:
        instance = instances_by_name[name]
        scenarios = ([replay] if replay is not None else
                     [generate_scenario(instance, rules=rules, mode="static"),
                      *(generate_scenario(instance, seed, rules=rules) for seed in scenario_seeds)])
        materialized: dict[str, Path] = {}
        for scenario in scenarios:
            scenario_path = output_dir / "scenarios" / f"{name}_{scenario.mode}_{scenario.seed}.json"
            if scenario_path.exists():
                saved = Scenario.load(scenario_path, instance)
                if saved.sha256 != scenario.sha256:
                    raise RuntimeError(f"Existing scenario differs; choose another output directory: {scenario_path}")
            else:
                scenario.save(scenario_path)
            materialized[scenario.sha256] = scenario_path
        instance_policies = [(policy.name, policy) for policy in policies]
        instance_policies.extend((method, None) for method in methods if method in {"edd", "greedy"})
        for policy_name, policy in instance_policies:
            for scenario in scenarios:
                mode, scenario_seed = scenario.mode, scenario.seed
                episode_id = _episode_id(
                    instance.name, policy_name, policy.training_seed if policy else None,
                    mode, scenario_seed
                )
                result, events = run_episode(
                    instance,
                    scenario.release_times,
                    policy_name,
                    policy=policy,
                    scenario_mode=mode,
                    scenario_seed=scenario_seed,
                    episode_id=episode_id,
                    scenario=scenario,
                )
                result["split"] = expected_split
                result["family"] = family(instance.name)
                result_rows.append(result)
                _json_line(results_path, result)
                _json_line(
                    manifest_path,
                    {
                        "episode_id": episode_id,
                        "instance": instance.name.upper(),
                        "split": expected_split,
                        "policy": policy_name,
                        "training_seed": policy.training_seed if policy else None,
                        "expression": result["expression"],
                        "expression_sha256": policy.expression_sha256 if policy else None,
                        "checkpoint_sha256": policy.checkpoint_sha256 if policy else None,
                        "scenario_mode": mode,
                        "scenario_seed": scenario_seed,
                        "simulator_version": manifest["simulator_version"],
                        "release_times": result["release_times"],
                        "scenario_file": str(materialized[scenario.sha256].resolve()),
                        "scenario_sha256": scenario.sha256,
                        "instance_sha256": instance.sha256,
                    },
                )
                _json_line(events_path, {"episode_id": episode_id, "events": events})
                episode_count += 1
                print(
                    f"[{phase} {episode_count}] {instance.name.upper()} {policy_name} "
                    f"seed={policy.training_seed if policy else '-'} {mode}/{scenario_seed}: "
                    f"served={result['served']}/{result['requested']} "
                    f"cancel={result['cancelled']} pending={result['unserved']} "
                    f"on_time={result['on_time_rate']:.3f} dist={result['distance']:.2f} "
                    f"violations={result['violation_count']} "
                    f"callback_p95={result['callback_latency_p95_ms']:.2f}ms",
                    flush=True,
                )
                _write_csv(episodes_path, [_csv_row(row) for row in result_rows])

    summary = _summarize(result_rows)
    comparisons = _paired_comparisons(result_rows)
    _write_csv(output_dir / "summary_by_policy.csv", summary)
    _write_csv(output_dir / "paired_comparisons.csv", comparisons)

    invalid = [row for row in result_rows if row["violation_count"] != 0]
    gate = {
        "phase": phase,
        "episodes": len(result_rows),
        "all_episodes_constraint_clean": not invalid,
        "episodes_with_violations": [
            {
                "episode_id": row["episode_id"],
                "violation_count": row["violation_count"],
                "violations": row["violations_json"],
            }
            for row in invalid
        ],
        "unserved_orders": sum(int(row["unserved"]) for row in result_rows),
        "mean_on_time_rate": statistics.fmean(float(row["on_time_rate"]) for row in result_rows),
        "mean_callback_latency_p95_ms": statistics.fmean(
            float(row["callback_latency_p95_ms"]) for row in result_rows
        ),
    }
    (output_dir / "gate_report.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if invalid:
        raise RuntimeError(
            f"{len(invalid)} {phase} episodes contained audited violations; "
            f"see {output_dir / 'gate_report.json'}"
        )
    return result_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zero-shot dynamic Solomon VRPTW evaluation of validation-selected GPRT/GP heuristics."
    )
    parser.add_argument("--phase", choices=("smoke", "pilot", "final", "all"), default="pilot")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--methods", nargs="+", choices=("gprt", "gp", "edd", "greedy"))
    parser.add_argument("--training-seeds", nargs="+", type=int)
    parser.add_argument("--policy-root", type=Path)
    parser.add_argument("--scenario-file", type=Path, help="Replay a saved scenario in --phase smoke")
    args = parser.parse_args()

    if args.scenario_file and args.phase != "smoke":
        parser.error("--scenario-file requires --phase smoke")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.methods:
        config["methods"] = args.methods
    if args.training_seeds:
        config["training_seeds"] = args.training_seeds
    if args.policy_root:
        config["policy_root"] = str(args.policy_root)
        config.pop("policy_roots", None)
    if version("dvrpsim") != config["dvrpsim_version"]:
        raise RuntimeError(
            f"Expected dvrpsim=={config['dvrpsim_version']}, found {version('dvrpsim')}"
        )
    if args.phase in {"pilot", "all"}:
        _run_phase(config, "pilot", args.output_dir / "pilot", config_path=args.config)
    if args.phase in {"final", "all"}:
        _run_phase(config, "final", args.output_dir / "final", config_path=args.config)
    if args.phase == "smoke":
        _run_phase(config, "smoke", args.output_dir / "smoke", config_path=args.config,
                   scenario_file=args.scenario_file)


if __name__ == "__main__":
    main()

"""Run paired train/validation-only GPRT jobs to measure routing acceleration."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--transformer-samples", type=int, default=8)
    parser.add_argument("--gp-generations", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/diagnostics/gprt_routing_diagnostic"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"seed": args.seed, "cycles": args.cycles, "split": "32 train / 12 validation; test skipped",
              "purpose": "performance diagnostic; main research preset is unchanged",
              "population": args.population, "transformer_samples": args.transformer_samples,
              "gp_generations": args.gp_generations,
              "threads": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
              "runs": {}}
    for backend in ("python", "numba"):
        directory = output / backend
        command = [sys.executable, str(root / "run_gprt_experiment.py"), "--config", str(root / "configs/gprt_main.json"),
                   "--cycles", str(args.cycles), "--seeds", str(args.seed), "--screening-run", "--skip-test",
                   "--population", str(args.population), "--transformer-samples", str(args.transformer_samples),
                   "--gp-generations", str(args.gp_generations),
                   "--route-backend", backend, "--output-dir", str(directory)]
        print(f"Starting {backend}: {args.cycles} cycles, train/validation only", flush=True)
        started = time.perf_counter()
        with (output / f"{backend}.log").open("w", encoding="utf-8") as stream:
            subprocess.run(command, cwd=root, env={**os.environ, **report["threads"]},
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        wall = time.perf_counter() - started
        results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        report["runs"][backend] = {"process_wall_seconds": wall, **results["searches"][0]}
        print(f"Finished {backend}: {wall:.3f}s", flush=True)
        (output / "progress.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    reference, fast = (report["runs"][mode] for mode in ("python", "numba"))
    for key in ("expression", "validation_fitness", "fitness_requests", "instance_evaluations",
                "unique_evaluations", "train_instance_requests", "validation_instance_requests", "cache_hits"):
        if reference[key] != fast[key]:
            raise AssertionError(f"Training result or evaluation budget differs: {key}")
    checkpoints = {}
    for backend in ("python", "numba"):
        directory = output / backend / "gprt" / str(args.seed)
        checkpoints[backend] = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    for key, tensor in checkpoints["python"]["model"].items():
        if not torch.equal(tensor, checkpoints["numba"]["model"][key]):
            raise AssertionError(f"Final model tensor differs: {key}")
    fields = ("batch_instances", "loss", "policy_loss", "elite_nll", "entropy", "train_fitness",
              "validation_fitness", "best_expression", "fitness_requests", "unique_evaluations")
    for left, right in zip(checkpoints["python"]["logs"], checkpoints["numba"]["logs"]):
        for key in fields:
            if left[key] != right[key]:
                raise AssertionError(f"Learning trajectory differs: cycle {left['cycle']}, {key}")
    report.update(identical_final_model_tensors=True, identical_trajectory=True, identical_budget=True,
                  end_to_end_speedup=reference["process_wall_seconds"] / fast["process_wall_seconds"],
                  evaluator_speedup=reference["evaluator_seconds"] / fast["evaluator_seconds"])
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

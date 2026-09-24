"""Read-only generator diagnostics; no new VRPTW evaluations or model updates."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from gprt_vrptw.config import GPRTConfig
from gprt_vrptw.grammar import CONSTANT_TOKENS, ARITY, Grammar, prefix_to_tree
from gprt_vrptw.trainer import seed_everything
from gprt_vrptw.transformer import CausalTransformer


def measure(model, trees):
    model.eval()
    grammar = model.grammar
    with torch.no_grad():
        log_probs = model.teacher_forced(trees)[0]
        logits = model(torch.tensor([[grammar.bos_id]]))[0, 0] / model.config.temperature
        valid = grammar.valid_tokens(grammar.initial_state())
        ids = [grammar.token_to_id[t] for t in valid]
        probabilities = torch.softmax(logits[ids], dim=0)
    fifo_roots = set(CONSTANT_TOKENS) | {
        "remaining_capacity", "route_load", "remaining_fraction", "current_time", "remaining_vehicles"
    }
    return {
        "probability_one_token_program": sum(float(p) for t, p in zip(valid, probabilities) if t not in ARITY),
        "probability_fifo_equivalent_one_token_program": sum(float(p) for t, p in zip(valid, probabilities) if t in fifo_roots),
        "screened_expression_log_probabilities": log_probs.tolist(),
    }


def run(root: Path):
    archive = json.loads((root / "baseline/20260914/candidate_archive.json").read_text())
    screened = sorted((e for e in archive if e["screening_fitness"] is not None),
                      key=lambda e: (e["screening_fitness"], e["expression"]))[:3]
    trees = [prefix_to_tree(e["tokens"]) for e in screened]
    rows = []
    for variant in ("baseline", "routing_rank", "elite3"):
        for seed in (20260914, 20260915):
            folder = root / variant / str(seed)
            if not (folder / "quality_summary.json").exists():
                continue
            state = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
            config = GPRTConfig(**state["config"])
            grammar = Grammar(config.max_tokens, config.max_depth, typed=config.typed_grammar,
                              include_lookahead=config.use_lookahead_features)
            seed_everything(seed)
            model = CausalTransformer(grammar, config)
            before = measure(model, trees)
            model.load_state_dict(state["model"])
            after = measure(model, trees)
            rows.append({"variant": variant, "seed": seed, "cycles": state["training_step"],
                         "before": before, "after": after})
            del state, model
    result = {
        "scope": "Exact teacher-forced probabilities in eval mode (dropout disabled); not the dropout-marginalized training policy. No VRPTW solve or optimizer step.",
        "probe_selection": "Top three expressions by common TRAIN screening fitness in baseline seed 20260914 archive. No validation fitness used to select probes.",
        "probes": [{"expression": e["expression"], "screening_fitness": e["screening_fitness"]} for e in screened],
        "runs": rows,
        "limitations": "Probability of specific syntax and one-token programs is diagnostic, not a measure of overall solution quality. Other expressions may implement equivalent or better policies.",
    }
    (root / "neural_likelihood_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run(Path("outputs/gprt_vrptw/quality_screening_20260917"))

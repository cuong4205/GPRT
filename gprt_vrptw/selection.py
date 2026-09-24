from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

from vrptw_baselines.data import VRPTWInstance

from .evaluator import Fitness, FitnessEvaluator, behavior_fingerprint
from .grammar import Tree, tree_to_prefix, tree_to_string


Archive = dict[str, dict[str, Any]]


def instance_set_id(instances: list[VRPTWInstance]) -> str:
    """Stable identity for a fitness batch; never compare it as global quality."""
    payload = "\n".join(sorted(instance.sha256 for instance in instances))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def record_observation(
    archive: Archive,
    tree: Tree,
    fitness: Fitness,
    cycle: int,
    instances: list[VRPTWInstance],
) -> None:
    expression = tree_to_string(tree)
    batch_id = instance_set_id(instances)
    entry = archive.setdefault(
        expression,
        {
            "tree": tree,
            "tokens": tree_to_prefix(tree),
            "observations": {},
            "screening_fitness": None,
        },
    )
    current = entry["observations"].get(batch_id)
    if current is None or fitness < current["fitness"]:
        entry["observations"][batch_id] = {
            "fitness": fitness,
            "cycle": cycle,
            "instances": tuple(sorted(instance.name.upper() for instance in instances)),
        }


def quota_candidates(archive: Archive, limit: int) -> list[dict[str, Any]]:
    """Take a balanced quota from every observed batch before common screening."""
    by_batch: dict[str, list[tuple[Fitness, str, dict[str, Any]]]] = defaultdict(list)
    for expression, entry in archive.items():
        for batch_id, observation in entry["observations"].items():
            by_batch[batch_id].append((observation["fitness"], expression, entry))
    if not by_batch:
        return []
    quota = max(1, math.ceil(limit / len(by_batch)))
    selected: dict[str, dict[str, Any]] = {}
    for observations in by_batch.values():
        for _, expression, entry in sorted(observations, key=lambda item: (item[0], item[1]))[:quota]:
            selected.setdefault(expression, entry)
    if len(selected) < limit:
        pooled = sorted(
            (
                min(observation["fitness"] for observation in entry["observations"].values()),
                expression,
                entry,
            )
            for expression, entry in archive.items()
            if expression not in selected
        )
        for _, expression, entry in pooled[: limit - len(selected)]:
            selected[expression] = entry
    return list(selected.values())


def screen_archive(
    archive: Archive,
    evaluator: FitnessEvaluator,
    screening_instances: list[VRPTWInstance],
    limit: int,
) -> list[tuple[Fitness, dict[str, Any]]]:
    candidates = quota_candidates(archive, limit)
    scored: list[tuple[Fitness, dict[str, Any]]] = []
    screening_id = instance_set_id(screening_instances)
    for entry in candidates:
        fitness = evaluator.fitness(
            entry["tree"], screening_instances, phase="screening"
        )
        entry["screening_fitness"] = fitness
        entry["screening_batch_id"] = screening_id
        scored.append((fitness, entry))
    return sorted(
        scored, key=lambda item: (item[0], tree_to_string(item[1]["tree"]))
    )[:limit]


def select_archive_seeds(
    archive: Archive,
    evaluator: FitnessEvaluator,
    screening_instances: list[VRPTWInstance],
    behavior_probe: list[VRPTWInstance],
    count: int,
    *,
    excluded_expressions: set[str] | None = None,
) -> tuple[list[Tree], int]:
    """Select screened, behaviorally distinct old elites for the next GP batch.

    Fitness from an old training batch is never used for selection here. Eligible
    entries must already have been evaluated on the current run's fixed screening
    set. GP evaluates every returned tree again on the new batch.
    """
    if count <= 0:
        return [], 0
    excluded = excluded_expressions or set()
    screening_id = instance_set_id(screening_instances)
    eligible = [
        entry
        for expression, entry in archive.items()
        if expression not in excluded
        and entry.get("screening_fitness") is not None
        and entry.get("screening_batch_id") == screening_id
    ]
    eligible.sort(
        key=lambda entry: (
            entry["screening_fitness"],
            tree_to_string(entry["tree"]),
        )
    )

    selected: list[Tree] = []
    deferred: list[Tree] = []
    fingerprints: set[tuple[tuple[tuple[int, ...], ...], ...]] = set()
    for entry in eligible:
        tree = entry["tree"]
        fingerprint = behavior_fingerprint(tree, behavior_probe, evaluator)
        if fingerprint in fingerprints:
            deferred.append(tree)
            continue
        selected.append(tree)
        fingerprints.add(fingerprint)
        if len(selected) == count:
            return selected, len(eligible)
    for tree in deferred:
        selected.append(tree)
        if len(selected) == count:
            break
    return selected, len(eligible)

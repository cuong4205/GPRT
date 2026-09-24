from __future__ import annotations

import hashlib
import random
from collections import defaultdict

from vrptw_baselines.data import VRPTWInstance


def family(name: str) -> str:
    stem = name.lower().split("_")[0]
    letters = "".join(c for c in stem if c.isalpha())
    digits = "".join(c for c in stem if c.isdigit())
    return f"{letters.upper()}{digits[:1]}"


def split_solomon(instances: list[VRPTWInstance]) -> dict[str, str]:
    """Fixed deterministic 32/12/12 stratified Solomon split."""
    grouped: dict[str, list[VRPTWInstance]] = defaultdict(list)
    for instance in instances:
        grouped[family(instance.name)].append(instance)
    result: dict[str, str] = {}
    for items in grouped.values():
        ordered = sorted(
            items,
            key=lambda item: hashlib.sha256(item.name.lower().encode()).hexdigest(),
        )
        n_val = max(1, round(len(ordered) * 0.2))
        n_test = max(1, round(len(ordered) * 0.2))
        n_train = len(ordered) - n_val - n_test
        for index, instance in enumerate(ordered):
            value = "train" if index < n_train else "validation" if index < n_train + n_val else "test"
            result[instance.name.lower()] = value
    counts = {part: sum(v == part for v in result.values()) for part in ("train", "validation", "test")}
    if len(instances) == 56 and counts != {"train": 32, "validation": 12, "test": 12}:
        raise AssertionError(f"Unexpected Solomon split counts: {counts}")
    return result


def stratified_batches(
    instances: list[VRPTWInstance], batch_size: int, *, seed: int | None = None
) -> list[list[VRPTWInstance]]:
    grouped: dict[str, list[VRPTWInstance]] = defaultdict(list)
    for instance in sorted(instances, key=lambda x: x.name.lower()):
        grouped[family(instance.name)].append(instance)
    if seed is not None:
        rng = random.Random(seed)
        for items in grouped.values():
            rng.shuffle(items)
        family_order = sorted(grouped)
        rng.shuffle(family_order)
    else:
        family_order = sorted(grouped)
    ordered: list[VRPTWInstance] = []
    depth = 0
    while len(ordered) < len(instances):
        for key in family_order:
            if depth < len(grouped[key]):
                ordered.append(grouped[key][depth])
        depth += 1
    return [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]


def stratified_sample(
    instances: list[VRPTWInstance], size: int
) -> list[VRPTWInstance]:
    """Fixed, family-balanced screening subset independent of training seed."""
    ordered = [item for batch in stratified_batches(instances, 1) for item in batch]
    return ordered[: min(size, len(ordered))]

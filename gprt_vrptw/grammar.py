from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TypeAlias


Tree: TypeAlias = str | float | tuple

SPECIAL = ("PAD", "BOS", "EOS")
CONSTANT_TOKENS = ("-1", "-0.5", "0", "0.5", "1", "2")
CORE_FEATURES = (
    "d_current",
    "d_depot",
    "demand",
    "ready_time",
    "due_date",
    "service_time",
    "arrival_time",
    "slack",
    "wait_time",
    "remaining_capacity",
    "route_load",
    "remaining_fraction",
    "current_time",
)
LOOKAHEAD_FEATURES = (
    "remaining_vehicles",
    "next_feasible_fraction",
    "urgency_rank",
)
FEATURES = CORE_FEATURES + LOOKAHEAD_FEATURES
ARITY = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "pdiv": 2,
    "min": 2,
    "max": 2,
    "ge": 2,
    "le": 2,
    "and": 2,
    "or": 2,
    "if_else": 3,
}
NUMERIC_OPERATORS = {"add", "sub", "mul", "pdiv", "min", "max"}
BOOLEAN_OPERATORS = {"ge", "le", "and", "or"}
SIGNATURES = {
    **{op: ("number", ("number", "number")) for op in NUMERIC_OPERATORS},
    "ge": ("boolean", ("number", "number")),
    "le": ("boolean", ("number", "number")),
    "and": ("boolean", ("boolean", "boolean")),
    "or": ("boolean", ("boolean", "boolean")),
    "if_else": ("number", ("boolean", "number", "number")),
}
VOCAB = SPECIAL + CONSTANT_TOKENS + FEATURES + tuple(ARITY)
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCAB)}


def tree_size(tree: Tree) -> int:
    return 1 if not isinstance(tree, tuple) else 1 + sum(tree_size(c) for c in tree[1:])


def tree_depth(tree: Tree) -> int:
    return 0 if not isinstance(tree, tuple) else 1 + max(tree_depth(c) for c in tree[1:])


def tree_to_prefix(tree: Tree) -> list[str]:
    if isinstance(tree, str):
        if tree not in FEATURES:
            raise ValueError(f"Unknown terminal: {tree}")
        return [tree]
    if isinstance(tree, (int, float)):
        token = f"{float(tree):g}"
        if token not in CONSTANT_TOKENS:
            raise ValueError(f"Constant outside vocabulary: {tree}")
        return [token]
    op = str(tree[0])
    if op not in ARITY or len(tree) != ARITY[op] + 1:
        raise ValueError(f"Invalid operator node: {tree}")
    return [op] + [token for child in tree[1:] for token in tree_to_prefix(child)]


def prefix_to_tree(tokens: list[str] | tuple[str, ...]) -> Tree:
    if not tokens:
        raise ValueError("Empty prefix expression")

    def parse(index: int) -> tuple[Tree, int]:
        if index >= len(tokens):
            raise ValueError("Incomplete prefix expression")
        token = tokens[index]
        if token in FEATURES:
            return token, index + 1
        if token in CONSTANT_TOKENS:
            return float(token), index + 1
        if token not in ARITY:
            raise ValueError(f"Token is not part of the expression grammar: {token}")
        children: list[Tree] = []
        cursor = index + 1
        for _ in range(ARITY[token]):
            child, cursor = parse(cursor)
            children.append(child)
        return (token, *children), cursor

    tree, end = parse(0)
    if end != len(tokens):
        raise ValueError("Trailing tokens after complete prefix expression")
    return tree


def tree_to_string(tree: Tree) -> str:
    if isinstance(tree, str):
        return tree
    if isinstance(tree, (int, float)):
        return f"{float(tree):g}"
    return f"{tree[0]}({', '.join(tree_to_string(c) for c in tree[1:])})"


def simplify_tree(tree: Tree, *, typed: bool = False) -> Tree:
    """Apply semantics-preserving, vocabulary-safe simplifications."""
    if not isinstance(tree, tuple):
        return tree
    op = str(tree[0])
    children = tuple(simplify_tree(child, typed=typed) for child in tree[1:])
    if op in {"add", "sub"} and children[1] == 0.0:
        return children[0]
    if op in {"mul", "pdiv"} and children[1] == 1.0:
        return children[0]
    if op in {"min", "max"} and children[0] == children[1]:
        return children[0]
    if op == "sub" and children[0] == children[1]:
        return 0.0
    if not typed and op in {"ge", "le"} and children[0] == children[1]:
        return 1.0
    if (
        op == "if_else"
        and isinstance(children[0], (int, float))
        and children[0] in {0.0, 1.0}
    ):
        return children[1] if children[0] == 1.0 else children[2]
    candidate = (op, *children)
    if (not typed or op not in BOOLEAN_OPERATORS) and all(isinstance(child, (int, float)) for child in children):
        value = safe_evaluate(candidate, {})
        token = f"{float(value):g}"
        if token in CONSTANT_TOKENS:
            return float(value)
    return candidate


def safe_evaluate(tree: Tree, values: dict[str, float], limit: float = 1e6) -> float:
    if isinstance(tree, str):
        value = float(values[tree])
    elif isinstance(tree, (int, float)):
        value = float(tree)
    else:
        op = tree[0]
        args = [safe_evaluate(child, values, limit) for child in tree[1:]]
        if op == "add":
            value = args[0] + args[1]
        elif op == "sub":
            value = args[0] - args[1]
        elif op == "mul":
            value = args[0] * args[1]
        elif op == "pdiv":
            value = args[0] if abs(args[1]) < 1e-9 else args[0] / args[1]
        elif op == "min":
            value = min(args)
        elif op == "max":
            value = max(args)
        elif op == "ge":
            value = float(args[0] >= args[1])
        elif op == "le":
            value = float(args[0] <= args[1])
        elif op == "and":
            value = float(bool(args[0]) and bool(args[1]))
        elif op == "or":
            value = float(bool(args[0]) or bool(args[1]))
        elif op == "if_else":
            value = args[1] if bool(args[0]) else args[2]
        else:
            raise ValueError(f"Unknown operator: {op}")
    if not math.isfinite(value):
        return math.copysign(limit, value if not math.isnan(value) else 1.0)
    return max(-limit, min(limit, value))


@dataclass(slots=True)
class GrammarState:
    """Pending prefix slots, storing the depth of each slot (next slot is last)."""

    pending_depths: list[int]
    pending_types: list[str]
    expression_tokens: int = 0

    @property
    def complete(self) -> bool:
        return not self.pending_depths


class Grammar:
    def __init__(
        self,
        max_tokens: int = 63,
        max_depth: int = 6,
        *,
        typed: bool = False,
        include_lookahead: bool = True,
    ):
        self.max_tokens = max_tokens
        self.max_depth = max_depth
        self.typed = typed
        self.features = FEATURES if include_lookahead else CORE_FEATURES
        self.vocab = SPECIAL + CONSTANT_TOKENS + self.features + tuple(ARITY)
        self.token_to_id = {token: index for index, token in enumerate(self.vocab)}
        self.id_to_token = dict(enumerate(self.vocab))
        self.pad_id = self.token_to_id["PAD"]
        self.bos_id = self.token_to_id["BOS"]
        self.eos_id = self.token_to_id["EOS"]

    def initial_state(self, root_type: str = "number") -> GrammarState:
        if root_type not in {"number", "boolean"}:
            raise ValueError("root_type must be number or boolean")
        return GrammarState([0], [root_type if self.typed else "any"], 0)

    def valid_tokens(self, state: GrammarState) -> list[str]:
        if state.complete:
            return ["EOS"]
        remaining_after = self.max_tokens - state.expression_tokens - 1
        if remaining_after < 0:
            return []
        depth = state.pending_depths[-1]
        if depth > self.max_depth:
            return []
        required_type = state.pending_types[-1]
        base_pending_types = state.pending_types[:-1]
        candidates: list[str] = []
        for token in CONSTANT_TOKENS + self.features + tuple(ARITY):
            arity = ARITY.get(token, 0)
            return_type = SIGNATURES[token][0] if token in SIGNATURES else "number"
            if self.typed and return_type != required_type:
                continue
            if arity and depth >= self.max_depth:
                continue
            child_types = (
                SIGNATURES[token][1]
                if self.typed and token in SIGNATURES
                else ("any",) * arity
            )
            new_pending_types = base_pending_types + list(reversed(child_types))
            new_pending_depths = state.pending_depths[:-1] + [depth + 1] * arity
            # A boolean has no terminal: its shortest tree is a comparison
            # with two numeric leaves, requiring one additional depth level.
            if any(
                slot_depth + (slot_type == "boolean") > self.max_depth
                for slot_depth, slot_type in zip(new_pending_depths, new_pending_types)
            ):
                continue
            min_tokens_needed = sum(
                3 if pending_type == "boolean" else 1
                for pending_type in new_pending_types
            )
            if min_tokens_needed <= remaining_after:
                candidates.append(token)
        return candidates

    def valid_token_ids(self, state: GrammarState) -> list[int]:
        return [self.token_to_id[token] for token in self.valid_tokens(state)]

    def step(self, state: GrammarState, token: str) -> None:
        if token not in self.valid_tokens(state):
            raise ValueError(f"Invalid grammar action {token!r} for state {state}")
        if token == "EOS":
            return
        depth = state.pending_depths.pop()
        state.pending_types.pop()
        arity = ARITY.get(token, 0)
        child_types = SIGNATURES[token][1] if self.typed and token in SIGNATURES else ("any",) * arity
        # Prefix decoding fills the first child next, while stacks pop from the end.
        state.pending_depths.extend([depth + 1] * arity)
        state.pending_types.extend(reversed(child_types))
        state.expression_tokens += 1

    def validate_expression(self, tokens: list[str] | tuple[str, ...]) -> bool:
        if not tokens or len(tokens) > self.max_tokens:
            return False
        state = self.initial_state()
        try:
            for token in tokens:
                self.step(state, token)
            if not state.complete:
                return False
            tree = prefix_to_tree(tokens)
            return tree_depth(tree) <= self.max_depth
        except (KeyError, ValueError):
            return False

    def random_tree(self, rng: random.Random, *, grow: bool = True, root_type: str = "number") -> Tree:
        state = self.initial_state(root_type)
        tokens: list[str] = []
        while not state.complete:
            valid = self.valid_tokens(state)
            if not valid:
                raise ValueError("No tree can complete the requested type within the grammar budget")
            terminals = [t for t in valid if t not in ARITY]
            operators = [t for t in valid if t in ARITY]
            choose_terminal = grow and terminals and rng.random() < 0.35
            token = rng.choice(terminals if choose_terminal or not operators else operators)
            self.step(state, token)
            tokens.append(token)
        return prefix_to_tree(tokens)

    def encode_expression(self, tree: Tree, include_special: bool = True) -> list[int]:
        tokens = tree_to_prefix(tree)
        if not self.validate_expression(tokens):
            raise ValueError("Tree violates grammar limits")
        ids = [self.token_to_id[t] for t in tokens]
        return [self.bos_id, *ids, self.eos_id] if include_special else ids

    def decode_expression(self, ids: list[int] | tuple[int, ...]) -> Tree:
        tokens = [self.id_to_token[i] for i in ids]
        tokens = [t for t in tokens if t not in {"PAD", "BOS", "EOS"}]
        return prefix_to_tree(tokens)

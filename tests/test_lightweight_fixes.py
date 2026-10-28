from __future__ import annotations

import random
import unittest
from pathlib import Path

from test_gprt_vrptw import tiny_instance
from gprt_vrptw.evaluator import (
    _next_feasible_fractions,
    _vectorized_next_feasible_fractions,
    _urgency_ranks,
    compile_tree,
    make_priority,
    normalized_feature_vector,
    normalized_features,
)
from gprt_vrptw.gp import crossover, deduplicate_fill, mutate, valid
from gprt_vrptw.grammar import FEATURES, Grammar
from vrptw_baselines.data import parse_instance
from vrptw_baselines.solver import CandidateContext, _candidate_contexts, solve


class LazyFeatureTests(unittest.TestCase):
    def test_batched_lookahead_features_match_dense_reference(self):
        instance = tiny_instance()
        contexts = _candidate_contexts(
            instance,
            set(range(1, len(instance.customers))),
            0,
            max(0.0, instance.depot.ready_time),
            instance.capacity,
            1,
        )
        fractions = _next_feasible_fractions(instance, contexts)
        ranks = _urgency_ranks(instance, contexts)
        tree = ("add", "next_feasible_fraction", "urgency_rank")
        priority = make_priority(tree)
        priority.prepare_contexts(instance, contexts)
        compiled = compile_tree(tree)
        for context in contexts:
            dense = normalized_features(instance, context)
            self.assertEqual(fractions[id(context)], dense["next_feasible_fraction"])
            self.assertEqual(ranks[id(context)], dense["urgency_rank"])
            self.assertEqual(
                priority(instance, context),
                compiled(normalized_feature_vector(instance, context)),
            )

    def test_vectorized_lookahead_matches_dense_reference(self):
        instance = parse_instance(Path("data/data/Solomon/c101.txt"))
        contexts = _candidate_contexts(
            instance,
            set(range(1, len(instance.customers))),
            0,
            max(0.0, instance.depot.ready_time),
            instance.capacity,
            1,
        )
        self.assertGreaterEqual(len(contexts) * len(contexts[0].unserved_ids), 2048)
        fractions = _vectorized_next_feasible_fractions(
            instance, contexts, instance._distance_matrix_cache
        )
        for context in contexts:
            self.assertEqual(
                fractions[id(context)],
                normalized_features(instance, context)["next_feasible_fraction"],
            )

    def test_all_terminals_and_expressions_match_dense_reference(self):
        instance = tiny_instance()
        rng = random.Random(281)
        grammar = Grammar(15, 3)
        trees = list(FEATURES) + [grammar.random_tree(rng) for _ in range(40)]
        trees.append(("if_else", ("le", "slack", "wait_time"), "urgency_rank", "next_feasible_fraction"))
        for _ in range(25):
            candidate = rng.randrange(1, 4)
            arrival = rng.uniform(0, 30)
            context = CandidateContext(
                candidate, rng.randrange(4), rng.uniform(0, 20), arrival,
                max(arrival, instance.customers[candidate].ready_time),
                rng.uniform(0, 2), 3, (1, 2, 3), rng.randrange(1, 4),
            )
            dense = normalized_feature_vector(instance, context)
            for tree in trees:
                self.assertEqual(
                    make_priority(tree)(instance, context),
                    compile_tree(tree)(dense),
                )
        for tree in trees:
            compiled = compile_tree(tree)
            reference = lambda inst, ctx: compiled(normalized_feature_vector(inst, ctx))
            self.assertEqual(
                solve(instance, "gp", priority_function=reference),
                solve(instance, "gp", priority_function=make_priority(tree)),
            )

    def test_unused_lookahead_does_not_scan_remaining_customers(self):
        class NoIteration:
            def __iter__(self):
                raise AssertionError("Unused look-ahead scanned remaining customers")

        instance = tiny_instance()
        context = CandidateContext(1, 0, 0, 1, 1, 2, 3, NoIteration())
        self.assertGreater(make_priority("d_current")(instance, context), 0)
        self.assertEqual(make_priority(1.0)(instance, context), 1)

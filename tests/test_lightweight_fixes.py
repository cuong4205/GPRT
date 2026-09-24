from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from test_gprt_vrptw import tiny_config, tiny_instance
from gprt_vrptw.evaluator import compile_tree, make_priority, normalized_feature_vector
from gprt_vrptw.gp import crossover, deduplicate_fill, mutate, valid
from gprt_vrptw.grammar import FEATURES, Grammar, GrammarState, safe_evaluate, simplify_tree, tree_to_prefix
from gprt_vrptw.trainer import GPRTTrainer
from vrptw_baselines.solver import CandidateContext, solve


class LazyFeatureTests(unittest.TestCase):
    def test_all_terminals_and_expressions_match_dense_reference(self):
        instance = tiny_instance()
        rng = random.Random(281)
        grammar = Grammar(15, 3)
        trees = list(FEATURES) + [grammar.random_tree(rng) for _ in range(40)]
        trees.append(("if_else", ("le", "slack", "wait_time"), "urgency_rank", "next_feasible_fraction"))
        for lookahead in (False, True):
            for _ in range(25):
                candidate = rng.randrange(1, 4)
                arrival = rng.uniform(0, 30)
                context = CandidateContext(
                    candidate, rng.randrange(4), rng.uniform(0, 20), arrival,
                    max(arrival, instance.customers[candidate].ready_time),
                    rng.uniform(0, 2), 3, (1, 2, 3), rng.randrange(1, 4),
                )
                dense = normalized_feature_vector(instance, context, use_lookahead_features=lookahead)
                for tree in trees:
                    self.assertEqual(
                        make_priority(tree, use_lookahead_features=lookahead)(instance, context),
                        compile_tree(tree)(dense),
                    )
            for tree in trees:
                compiled = compile_tree(tree)
                def reference(inst, ctx):
                    return compiled(normalized_feature_vector(inst, ctx, use_lookahead_features=lookahead))
                self.assertEqual(
                    solve(instance, "gp", priority_function=reference),
                    solve(instance, "gp", priority_function=make_priority(tree, use_lookahead_features=lookahead)),
                )

    def test_unused_lookahead_does_not_scan_remaining_customers(self):
        class NoIteration:
            def __iter__(self):
                raise AssertionError("Unused look-ahead scanned remaining customers")

        instance = tiny_instance()
        context = CandidateContext(1, 0, 0, 1, 1, 2, 3, NoIteration())
        self.assertGreater(make_priority("d_current")(instance, context), 0)
        self.assertEqual(make_priority("urgency_rank", use_lookahead_features=False)(instance, context), 0)
        self.assertEqual(make_priority(1.0)(instance, context), 1)


class TypedGrammarRegressionTests(unittest.TestCase):
    def test_depth_boundary_cannot_open_impossible_boolean_slot(self):
        grammar = Grammar(63, 6, typed=True)
        state = grammar.initial_state()
        for _ in range(5):
            grammar.step(state, "add")
        self.assertNotIn("if_else", grammar.valid_tokens(state))
        self.assertNotIn("if_else", Grammar(63, 1, typed=True).valid_tokens(Grammar(63, 1, typed=True).initial_state()))

    def test_all_reachable_small_grammar_states_have_completion(self):
        # Exhaust states, merging prefixes with the same pending slots. Every
        # permitted step must have a successor; token count makes this acyclic.
        for depth in range(4):
            for budget in range(1, 10):
                grammar = Grammar(budget, depth, typed=True)
                frontier = [grammar.initial_state()]
                seen = set()
                while frontier:
                    state = frontier.pop()
                    key = (tuple(state.pending_depths), tuple(state.pending_types), state.expression_tokens)
                    if key in seen:
                        continue
                    seen.add(key)
                    tokens = grammar.valid_tokens(state)
                    self.assertTrue(tokens, (depth, budget, key))
                    if state.complete:
                        self.assertEqual(tokens, ["EOS"])
                        continue
                    for token in tokens:
                        child = GrammarState(state.pending_depths.copy(), state.pending_types.copy(), state.expression_tokens)
                        grammar.step(child, token)
                        frontier.append(child)

    def test_simplification_and_gp_keep_types_and_semantics(self):
        grammar = Grammar(31, 4, typed=True)
        rng = random.Random(41)
        original = ("if_else", ("and", ("le", "d_current", "d_current"), ("le", "demand", "remaining_capacity")), "d_current", "due_date")
        trees = [original] + [grammar.random_tree(rng) for _ in range(100)]
        values = {name: rng.uniform(-3, 3) for name in FEATURES}
        for tree in trees:
            simplified = simplify_tree(tree, typed=True)
            self.assertTrue(valid(simplified, grammar))
            self.assertEqual(safe_evaluate(tree, values), safe_evaluate(simplified, values))
            self.assertTrue(valid(mutate(tree, rng, grammar), grammar))
            self.assertTrue(valid(crossover(tree, original, rng, grammar), grammar))
        self.assertTrue(all(valid(tree, grammar) for tree in deduplicate_fill(trees[:8], 16, rng, grammar)))
        boolean = Grammar(3, 1, typed=True).random_tree(rng, root_type="boolean")
        self.assertIn(boolean[0], {"le", "ge"})

    def test_typed_training_runs_through_sampling_gp_and_imitation(self):
        config = tiny_config(cycles=1)
        config.typed_grammar = True
        config.max_tokens = 15
        config.max_depth = 3
        with tempfile.TemporaryDirectory() as directory:
            trainer = GPRTTrainer(config)
            result = trainer.train(
                [tiny_instance("train", "typed-train")],
                [tiny_instance("val", "typed-val")],
                seed=41, output_dir=Path(directory),
                split_manifest={"train": "train", "val": "validation"},
            )
            self.assertIsNotNone(result.best_tree)
            self.assertTrue(trainer.grammar.validate_expression(tree_to_prefix(result.best_tree)))


if __name__ == "__main__":
    unittest.main()

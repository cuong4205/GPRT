from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

import torch

from run_gprt_experiment import current_process_rss_bytes
from gprt_vrptw.config import GPRTConfig
from gprt_vrptw.evaluator import compile_tree, normalized_features
from gprt_vrptw.gp import crossover, mutate, valid
from gprt_vrptw.grammar import (
    ARITY,
    FEATURES,
    Grammar,
    prefix_to_tree,
    safe_evaluate,
    tree_depth,
    tree_to_prefix,
)
from gprt_vrptw.selection import (
    quota_candidates,
    record_observation,
    screen_archive,
    select_archive_seeds,
)
from gprt_vrptw.splits import split_solomon
from gprt_vrptw.trainer import GPRTTrainer, normalized_advantages, seed_everything
from gprt_vrptw.transformer import CausalTransformer
from vrptw_baselines.data import Customer, VRPTWInstance, discover_instances
from vrptw_baselines.solver import CandidateContext, validate_solution


def tiny_instance(name: str = "tiny", sha: str = "tiny") -> VRPTWInstance:
    return VRPTWInstance(
        name=name,
        path=Path(f"{name}.txt"),
        max_vehicles=2,
        capacity=2,
        customers=(
            Customer(0, 0, 0, 0, 0, 100, 0),
            Customer(1, 1, 0, 1, 0, 20, 1),
            Customer(2, 2, 0, 1, 5, 20, 1),
            Customer(3, 8, 0, 1, 0, 20, 1),
        ),
        sha256=sha,
        source="synthetic",
    )


def tiny_config(cycles: int = 2) -> GPRTConfig:
    return GPRTConfig(
        d_model=16,
        nhead=4,
        num_layers=1,
        dim_feedforward=32,
        dropout=0.1,
        max_tokens=7,
        max_depth=2,
        population_size=4,
        transformer_samples=2,
        tournament_size=2,
        elite_count=1,
        gp_generations=1,
        cycles=cycles,
        batch_size_instances=1,
        validation_interval=2,
        validation_candidates=2,
    )


class GrammarTests(unittest.TestCase):
    def test_tree_prefix_round_trip_and_semantics(self) -> None:
        tree = ("if_else", ("ge", "slack", 0.0), ("add", "d_current", 0.5), ("pdiv", "d_depot", 2.0))
        tokens = tree_to_prefix(tree)
        restored = prefix_to_tree(tokens)
        self.assertEqual(restored, tree)
        values = {name: 0.25 for name in Grammar().vocab if name not in ARITY}
        self.assertEqual(safe_evaluate(tree, values), safe_evaluate(restored, values))

    def test_arity_and_grammar_mask(self) -> None:
        grammar = Grammar(max_tokens=7, max_depth=2)
        state = grammar.initial_state()
        self.assertNotIn("EOS", grammar.valid_tokens(state))
        grammar.step(state, "add")
        self.assertEqual(len(state.pending_depths), 2)
        grammar.step(state, "d_current")
        self.assertEqual(len(state.pending_depths), 1)
        grammar.step(state, "d_depot")
        self.assertEqual(grammar.valid_tokens(state), ["EOS"])
        with self.assertRaises(ValueError):
            prefix_to_tree(["add", "d_current"])

    def test_depth_and_length_limits(self) -> None:
        grammar = Grammar(max_tokens=3, max_depth=1)
        self.assertTrue(grammar.validate_expression(["add", "d_current", "d_depot"]))
        self.assertFalse(grammar.validate_expression(["add", "add", "demand", "demand", "demand"]))
        state = grammar.initial_state()
        grammar.step(state, "add")
        self.assertTrue(all(token not in ARITY for token in grammar.valid_tokens(state)))

    def test_protected_division_and_logic(self) -> None:
        values = {"demand": 3.0, "slack": -1.0}
        self.assertEqual(safe_evaluate(("pdiv", "demand", 0.0), values), 3.0)
        self.assertEqual(safe_evaluate(("ge", "demand", 2.0), values), 1.0)
        self.assertEqual(safe_evaluate(("le", "demand", 2.0), values), 0.0)
        self.assertEqual(safe_evaluate(("and", 1.0, 0.0), values), 0.0)
        self.assertEqual(safe_evaluate(("or", 0.0, -1.0), values), 1.0)
        self.assertEqual(safe_evaluate(("if_else", 1.0, 2.0, -1.0), values), 2.0)

    def test_gp_operators_remain_valid(self) -> None:
        grammar = Grammar(max_tokens=15, max_depth=3)
        rng = random.Random(9)
        for _ in range(50):
            a, b = grammar.random_tree(rng), grammar.random_tree(rng)
            self.assertTrue(valid(crossover(a, b, rng, grammar), grammar))
            self.assertTrue(valid(mutate(a, rng, grammar), grammar))

    def test_compiled_evaluator_matches_tree_interpreter(self) -> None:
        grammar = Grammar(max_tokens=31, max_depth=4)
        rng = random.Random(91)
        values = {name: rng.uniform(-2, 2) for name in grammar.vocab if name not in ARITY}
        vector = tuple(values[name] for name in FEATURES)
        for _ in range(100):
            tree = grammar.random_tree(rng)
            self.assertAlmostEqual(compile_tree(tree)(vector), safe_evaluate(tree, values), places=10)

    def test_typed_grammar_keeps_boolean_nodes_in_boolean_slots(self) -> None:
        grammar = Grammar(max_tokens=7, max_depth=2, typed=True)
        state = grammar.initial_state()
        self.assertNotIn("ge", grammar.valid_tokens(state))
        grammar.step(state, "if_else")
        self.assertIn("ge", grammar.valid_tokens(state))
        self.assertNotIn("d_current", grammar.valid_tokens(state))
        grammar.step(state, "ge")
        self.assertIn("d_current", grammar.valid_tokens(state))
        rng = random.Random(123)
        for _ in range(30):
            self.assertTrue(grammar.validate_expression(tree_to_prefix(grammar.random_tree(rng))))


class TransformerTests(unittest.TestCase):
    def setUp(self) -> None:
        seed_everything(123)
        self.config = tiny_config()
        self.grammar = Grammar(self.config.max_tokens, self.config.max_depth)
        self.model = CausalTransformer(self.grammar, self.config)
        self.model.eval()

    def test_causal_mask_blocks_future(self) -> None:
        a = torch.tensor([[self.grammar.bos_id, self.grammar.token_to_id["add"], self.grammar.token_to_id["d_current"]]])
        b = torch.tensor([[self.grammar.bos_id, self.grammar.token_to_id["add"], self.grammar.token_to_id["d_depot"]]])
        with torch.no_grad():
            logits_a, logits_b = self.model(a), self.model(b)
        self.assertTrue(torch.equal(logits_a[:, :2], logits_b[:, :2]))

    def test_padding_mask_is_independent(self) -> None:
        short = torch.tensor([[self.grammar.bos_id, self.grammar.token_to_id["d_current"]]])
        padded = torch.tensor([[self.grammar.bos_id, self.grammar.token_to_id["d_current"], self.grammar.pad_id]])
        with torch.no_grad():
            a = self.model(short)
            b = self.model(padded, padded.eq(self.grammar.pad_id))
        self.assertTrue(torch.allclose(a, b[:, :2], atol=1e-6, rtol=0))

    def test_sampling_always_valid_and_bounded(self) -> None:
        with torch.no_grad():
            samples = self.model.generate(40)
        for sample in samples:
            tokens = [self.grammar.id_to_token[i] for i in sample.token_ids]
            self.assertTrue(self.grammar.validate_expression(tokens))
            self.assertLessEqual(len(tokens), self.config.max_tokens)
            self.assertLessEqual(tree_depth(sample.tree), self.config.max_depth)

    def test_teacher_forced_log_probability(self) -> None:
        trees = ["d_current", ("add", "d_current", 1.0)]
        log_probs, nll, entropy = self.model.teacher_forced(trees)
        self.assertEqual(tuple(log_probs.shape), (2,))
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertGreater(float(nll.detach()), 0)
        self.assertGreater(float(entropy.detach()), 0)

    def test_teacher_forcing_matches_sampling_temperature(self) -> None:
        self.config.temperature = 0.5
        with torch.no_grad():
            sample = self.model.generate(1, greedy=True)[0]
            recomputed = self.model.teacher_forced([sample.tree])[0][0]
        self.assertTrue(torch.allclose(sample.log_probability, recomputed, atol=1e-6, rtol=0))

    def test_reinforce_toy_update_moves_probability(self) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        good, bad = "d_current", "d_depot"
        before = self.model.teacher_forced([good, bad])[0].detach()
        for _ in range(8):
            log_probs, _, _ = self.model.teacher_forced([good, bad])
            loss = -(torch.tensor([1.0, -1.0]) * log_probs).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        after = self.model.teacher_forced([good, bad])[0].detach()
        self.assertGreater(float(after[0] - after[1]), float(before[0] - before[1]))


class ProtocolTests(unittest.TestCase):
    def test_process_memory_measurement_is_available(self) -> None:
        self.assertGreater(current_process_rss_bytes(), 0)

    def test_feature_set_has_normalized_lookahead_features(self) -> None:
        instance = tiny_instance()
        context = CandidateContext(1, 0, 0.0, 1.0, 1.0, 2.0, 3)
        values = normalized_features(instance, context)
        self.assertEqual(len(values), 16)
        self.assertIn("service_time", values)
        self.assertIn("arrival_time", values)
        self.assertIn("remaining_vehicles", values)
        self.assertIn("next_feasible_fraction", values)
        self.assertIn("urgency_rank", values)
        grammar = Grammar(include_lookahead=False)
        self.assertNotIn("urgency_rank", grammar.vocab)

    def test_incomplete_train_coverage_is_rejected(self) -> None:
        config = tiny_config(cycles=1)
        train = [tiny_instance("a", "a"), tiny_instance("b", "b")]
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "covers only"):
                GPRTTrainer(config).train(
                    train,
                    [tiny_instance("val", "val")],
                    seed=1,
                    output_dir=Path(temp),
                    split_manifest={"a": "train", "b": "train", "val": "validation"},
                )

    def test_ewma_advantage_is_not_cancelled_by_centering(self) -> None:
        rewards = torch.tensor([0.0, 1.0])
        ewma = normalized_advantages(rewards, 0.25, "ewma_scale")
        standardized = normalized_advantages(rewards, 0.25, "batch_standardize")
        self.assertNotAlmostEqual(float(ewma.mean()), 0.0)
        self.assertAlmostEqual(float(standardized.mean()), 0.0)

    def test_archive_quota_includes_different_batches(self) -> None:
        from gprt_vrptw.evaluator import Fitness

        archive = {}
        easy, hard = tiny_instance("easy", "easy"), tiny_instance("hard", "hard")
        record_observation(archive, "d_current", Fitness(0, 0, 1, 1, 1), 1, [easy])
        record_observation(archive, "d_depot", Fitness(0, 0, 9, 9, 1), 2, [hard])
        chosen = quota_candidates(archive, 2)
        self.assertEqual({item["tree"] for item in chosen}, {"d_current", "d_depot"})

    def test_archive_seeds_require_common_screening_and_exclude_neural_duplicates(self) -> None:
        from gprt_vrptw.evaluator import Fitness, FitnessEvaluator

        archive = {}
        old_batch = tiny_instance("old", "old")
        screening = [tiny_instance("screen", "screen")]
        record_observation(archive, "d_current", Fitness(0, 0, 1, 1, 1), 1, [old_batch])
        record_observation(archive, "d_depot", Fitness(0, 0, 2, 2, 1), 1, [old_batch])
        evaluator = FitnessEvaluator()
        self.assertEqual(
            select_archive_seeds(archive, evaluator, screening, screening, 1)[0],
            [],
        )
        screen_archive(archive, evaluator, screening, 2)
        selected, pool_size = select_archive_seeds(
            archive,
            evaluator,
            screening,
            screening,
            1,
            excluded_expressions={"d_current"},
        )
        self.assertEqual(pool_size, 1)
        self.assertEqual(selected, ["d_depot"])

    def test_archive_seeding_replaces_random_seed_and_is_reevaluated(self) -> None:
        train = [tiny_instance("a", "a"), tiny_instance("b", "b")]
        validation = [tiny_instance("val", "val")]
        config = tiny_config(cycles=2)
        config.validation_interval = 1
        config.archive_seed_count = 1
        with tempfile.TemporaryDirectory() as temp:
            trainer = GPRTTrainer(config)
            trainer.train(
                train,
                validation,
                seed=17,
                output_dir=Path(temp),
                split_manifest={"a": "train", "b": "train", "val": "validation"},
            )
            second = trainer.logs[1]
            self.assertEqual(second["archive_seed_count"], 1)
            self.assertEqual(second["initial_seed_count"], config.population_size)
            self.assertEqual(
                second["neural_seed_count"]
                + second["archive_seed_count"]
                + second["random_seed_count"],
                config.population_size,
            )
            self.assertEqual(len(second["archive_seed_fitness"]), 1)

    def test_solution_checker_rejects_all_required_failures(self) -> None:
        instance = tiny_instance()
        self.assertFalse(validate_solution(instance, [(0, 1, 2, 3, 0)])["feasible"])  # overload
        checked = validate_solution(instance, [(0, 1, 1, 0), (0, 2, 0)])
        self.assertFalse(checked["feasible"])
        self.assertTrue(checked["duplicates"])
        self.assertTrue(checked["missing"])
        self.assertFalse(validate_solution(instance, [(1, 2, 0), (0, 3, 0)])["feasible"])  # wrong depot
        too_many = [(0, 1, 0), (0, 2, 0), (0, 3, 0)]
        self.assertFalse(validate_solution(instance, too_many)["feasible"])  # fleet

    def test_split_is_32_12_12_and_training_excludes_test(self) -> None:
        instances = discover_instances("data/data/Solomon")
        split = split_solomon(instances)
        self.assertEqual([sum(v == p for v in split.values()) for p in ("train", "validation", "test")], [32, 12, 12])
        self.assertTrue(set(k for k, v in split.items() if v == "train").isdisjoint(k for k, v in split.items() if v == "test"))

    def test_checkpoint_resume_is_exact(self) -> None:
        train = [tiny_instance("train", "train")]
        validation = [tiny_instance("val", "val")]
        manifest = {"train": "train", "val": "validation", "secret-test": "test"}
        config = tiny_config(cycles=2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            full = GPRTTrainer(config)
            result_full = full.train(train, validation, seed=77, output_dir=root / "full", split_manifest=manifest)
            interrupted = GPRTTrainer(config)
            interrupted.train(
                train,
                validation,
                seed=77,
                output_dir=root / "resume",
                split_manifest=manifest,
                stop_after_cycle=1,
            )
            resumed = GPRTTrainer(config)
            result_resumed = resumed.train(
                train,
                validation,
                seed=77,
                output_dir=root / "resume",
                split_manifest=manifest,
                resume=root / "resume" / "last.pt",
            )
            self.assertEqual(result_full.best_tree, result_resumed.best_tree)
            self.assertEqual(result_full.best_validation_fitness, result_resumed.best_validation_fitness)
            for key, value in full.model.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.model.state_dict()[key]), key)
            archive = json.loads((root / "resume" / "candidate_archive.json").read_text())
            self.assertTrue(archive)
            self.assertIn("observations", archive[0])
            self.assertIn("batch_id", archive[0]["observations"][0])
            summary = json.loads((root / "resume" / "training_summary.json").read_text())
            self.assertEqual(summary["training_coverage_rate"], 1.0)
            self.assertEqual(summary["screening_instance_names"], ["train"])

    def test_same_seed_reproduces_model(self) -> None:
        train = [tiny_instance("train", "train")]
        validation = [tiny_instance("val", "val")]
        config = tiny_config(cycles=1)
        with tempfile.TemporaryDirectory() as temp:
            states = []
            trees = []
            for index in range(2):
                trainer = GPRTTrainer(config)
                result = trainer.train(
                    train,
                    validation,
                    seed=88,
                    output_dir=Path(temp) / str(index),
                    split_manifest={"train": "train", "val": "validation"},
                )
                states.append({k: v.clone() for k, v in trainer.model.state_dict().items()})
                trees.append(result.best_tree)
            self.assertEqual(trees[0], trees[1])
            self.assertTrue(all(torch.equal(states[0][key], states[1][key]) for key in states[0]))

    def test_standalone_rl_uses_declared_fitness_budget(self) -> None:
        train = [tiny_instance("train", "train")]
        validation = [tiny_instance("val", "val")]
        config = tiny_config(cycles=1)
        config.use_gp = False
        config.use_elite_imitation = False
        config.standalone_rl_samples_per_cycle = 6
        with tempfile.TemporaryDirectory() as temp:
            trainer = GPRTTrainer(config)
            trainer.train(
                train,
                validation,
                seed=99,
                output_dir=Path(temp),
                split_manifest={"train": "train", "val": "validation"},
            )
            # Six policy rewards plus two common-screening and two validation requests.
            self.assertEqual(trainer.evaluator.stats.fitness_requests, 10)


if __name__ == "__main__":
    unittest.main()

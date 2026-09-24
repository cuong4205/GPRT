from contextlib import redirect_stdout
from dataclasses import asdict
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_gprt_vrptw import tiny_config, tiny_instance
from gprt_vrptw.evaluator import Fitness, rank_rewards
from gprt_vrptw.trainer import GPRTTrainer
from scripts.screen_quality_hypotheses import ReplayEvaluator, reward_diagnostic


class QualityScreeningTests(unittest.TestCase):
    def test_routing_rank_preserves_quality_order_and_groups_complexity_ties(self):
        values = [Fitness(0, 0, 2, 0.2, 0.01), Fitness(0, 0, 2, 0.2, 0.02),
                  Fitness(0, 0, 3, 0.1, 0.001), Fitness(1, 0.1, 1, 0.01, 0.001)]
        rewards = rank_rewards([f._replace(complexity=0.0) for f in values])
        self.assertEqual(rewards[0], rewards[1])
        self.assertGreater(rewards[1], rewards[2])
        self.assertGreater(rewards[2], rewards[3])
        self.assertEqual(reward_diagnostic(values)["pairs_separated_only_by_complexity"], 1)

    def test_replaying_solutions_preserves_training_parameters_and_logical_counts(self):
        config = tiny_config(cycles=2)
        train = [tiny_instance("train", "replay-train")]
        validation = [tiny_instance("val", "replay-val")]
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            first = GPRTTrainer(config)
            a = first.train(train, validation, seed=415, output_dir=Path(directory)/"cold",
                            split_manifest={"train": "train", "val": "validation"})
            cache = dict(first.evaluator.cache)
            with patch("gprt_vrptw.trainer.FitnessEvaluator",
                       lambda *args, **kwargs: ReplayEvaluator(*args, replay_cache=cache, **kwargs)):
                replay = GPRTTrainer(config)
                b = replay.train(train, validation, seed=415, output_dir=Path(directory)/"replay",
                                 split_manifest={"train": "train", "val": "validation"})
            self.assertEqual(a.best_tree, b.best_tree)
            self.assertEqual(a.best_validation_fitness, b.best_validation_fitness)
            for key, value in first.model.state_dict().items():
                self.assertTrue(torch.equal(value, replay.model.state_dict()[key]), key)
            stats_a, stats_b = asdict(first.evaluator.stats), asdict(replay.evaluator.stats)
            stats_a.pop("evaluator_seconds")
            stats_b.pop("evaluator_seconds")
            self.assertEqual(stats_a, stats_b)
            self.assertEqual(replay.evaluator.replay_hits, replay.evaluator.stats.instance_evaluations)
            self.assertEqual(replay.evaluator.stats.evaluator_seconds, 0.0)
            for left, right in zip(a.logs, b.logs):
                for key in ("loss", "policy_loss", "elite_nll", "train_fitness", "best_expression",
                            "gradient_norm", "neural_direct_elite_count"):
                    self.assertEqual(left[key], right[key], key)


if __name__ == "__main__":
    unittest.main()

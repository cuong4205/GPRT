from __future__ import annotations

import unittest
from math import hypot
from pathlib import Path

from vrptw_baselines.data import Customer, VRPTWInstance, parse_instance
from vrptw_baselines.solver import solve, validate_solution


class VRPTWBaselineTests(unittest.TestCase):
    def test_parse_solomon(self) -> None:
        instance = parse_instance(Path("data/data/Solomon/c101.txt"))
        self.assertEqual(instance.customer_count, 100)
        self.assertEqual(instance.max_vehicles, 25)
        self.assertEqual(instance.capacity, 200)
        self.assertEqual(instance.depot.customer_id, 0)

    def test_distance_matrix_matches_unrounded_euclidean_distance(self) -> None:
        instance = parse_instance(Path("data/data/Solomon/c101.txt"))
        for a in instance.customers:
            for b in instance.customers:
                self.assertEqual(
                    instance.distance(a.customer_id, b.customer_id),
                    hypot(a.x - b.x, a.y - b.y),
                )

    def tiny_instance(self) -> VRPTWInstance:
        return VRPTWInstance(
            name="tiny",
            path=Path("tiny.txt"),
            max_vehicles=2,
            capacity=2,
            customers=(
                Customer(0, 0, 0, 0, 0, 100, 0),
                Customer(1, 1, 0, 1, 0, 20, 1),
                Customer(2, 2, 0, 1, 5, 20, 1),
                Customer(3, 8, 0, 1, 0, 20, 1),
            ),
            sha256="test",
            source="synthetic",
        )

    def test_constructive_methods_are_feasible(self) -> None:
        instance = self.tiny_instance()
        for method in ("fifo", "random", "greedy", "edd"):
            solution = solve(instance, method, seed=7)
            self.assertTrue(solution.feasible, method)
            self.assertTrue(validate_solution(instance, solution.routes)["feasible"])

    def test_checker_detects_duplicate_and_missing_customer(self) -> None:
        checked = validate_solution(self.tiny_instance(), [(0, 1, 1, 0), (0, 2, 0)])
        self.assertFalse(checked["feasible"])
        self.assertIn(3, checked["missing"])
        self.assertIn(1, checked["duplicates"])

if __name__ == "__main__":
    unittest.main()

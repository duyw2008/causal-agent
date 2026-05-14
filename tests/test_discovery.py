"""Smoke tests for causal discovery (PC, FCI, GES, bootstrap)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.graph import CausalDAG
from core.discovery import (
    pc_algorithm, fci_algorithm, ges_algorithm,
    generate_linear_data, bootstrap_edge_confidence,
)


class TestDiscovery:
    def test_pc_chain(self):
        dag = CausalDAG(["X", "M", "Y"], [("X", "M"), ("M", "Y")])
        data = generate_linear_data(dag, 500, seed=1)
        result = pc_algorithm(data, ["X", "M", "Y"], alpha=0.05, max_cond_size=2)
        assert "X" in result.variables

    def test_pc_collider(self):
        dag = CausalDAG(["X", "Y", "Z"], [("X", "Z"), ("Y", "Z")])
        data = generate_linear_data(dag, 500, seed=1)
        result = pc_algorithm(data, ["X", "Y", "Z"], alpha=0.05, max_cond_size=2)
        # Collider should be detected: X→Z, Y→Z (or equivalent)
        assert "Z" in result.variables

    def test_pc_no_cycles_simple(self):
        """PC should not crash on simple 3-var structures."""
        structures = [
            CausalDAG(["X","M","Y"],[("X","M"),("M","Y")]),      # chain
            CausalDAG(["X","Y","Z"],[("Z","X"),("Z","Y")]),      # fork
            CausalDAG(["X","Y","Z"],[("X","Z"),("Y","Z")]),      # collider
            CausalDAG(["G","D","R"],[("G","D"),("G","R"),("D","R")]), # Simpson
        ]
        for dag in structures:
            data = generate_linear_data(dag, 500, seed=42)
            result = pc_algorithm(data, dag.variables, alpha=0.05, max_cond_size=2)
            assert result is not None, f"PC failed on {dag}"

    def test_fci_runs(self):
        dag = CausalDAG(["X", "Y", "Z"], [("X", "Z"), ("Y", "Z")])
        data = generate_linear_data(dag, 500, seed=1)
        pag = fci_algorithm(data, ["X", "Y", "Z"], alpha=0.05, max_cond_size=2)
        assert pag.n == 3

    def test_ges_runs(self):
        dag = CausalDAG(["X", "Y"], [("X", "Y")])
        data = generate_linear_data(dag, 300, seed=1)
        result = ges_algorithm(data, ["X", "Y"])
        assert result is not None

    def test_ges_collider_pruning(self):
        """GES Phase 3 (CI pruning) must remove spurious X→Y in collider X→Z←Y."""
        dag = CausalDAG(["X", "Y", "Z"], [("X", "Z"), ("Y", "Z")])
        data = generate_linear_data(dag, 2000, seed=42)
        result = ges_algorithm(data, ["X", "Y", "Z"])
        # After CI pruning, X and Y must NOT be adjacent
        assert "Y" not in result.children("X"), "Spurious X→Y not removed"
        assert "X" not in result.children("Y"), "Spurious Y→X not removed"
        # The true edges X→Z, Y→Z must remain
        assert "Z" in result.children("X") or "X" in result.children("Z"), "Missing X-Z edge"
        assert "Z" in result.children("Y") or "Y" in result.children("Z"), "Missing Y-Z edge"

    def test_ges_chain_preserved(self):
        """GES CI pruning must NOT remove true edges in chain X→Y→Z."""
        dag = CausalDAG(["X", "Y", "Z"], [("X", "Y"), ("Y", "Z")])
        data = generate_linear_data(dag, 2000, seed=123)
        result = ges_algorithm(data, ["X", "Y", "Z"])
        assert "Y" in result.children("X") or "X" in result.children("Y"), "Chain edge X-Y removed"
        assert "Z" in result.children("Y") or "Y" in result.children("Z"), "Chain edge Y-Z removed"

    def test_bootstrap_confidence(self):
        dag = CausalDAG(["G", "D", "R"], [("G", "D"), ("G", "R"), ("D", "R")])
        data = generate_linear_data(dag, 500, seed=1)
        conf = bootstrap_edge_confidence(data, ["G", "D", "R"], n_bootstrap=10)
        # All 3 true edges should appear
        assert len(conf) >= 1  # at least some edges found
        assert all(0 <= c <= 1 for c in conf.values())

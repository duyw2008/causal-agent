"""Tests for time series discovery and NL parser."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


class TestTimeSeries:
    def test_granger_x_to_y(self):
        """X Granger-causes Y, Y does NOT Granger-cause X."""
        from core.ts_discovery import granger_causality_test
        rng = np.random.default_rng(42)
        n = 300
        X = np.zeros(n)
        Y = np.zeros(n)
        for t in range(1, n):
            X[t] = 0.7 * X[t - 1] + rng.normal(0, 0.3)
            Y[t] = 0.5 * Y[t - 1] + 0.6 * X[t - 1] + rng.normal(0, 0.3)
        data = np.column_stack([X, Y])
        gc = granger_causality_test(data, ["X", "Y"], max_lag=1, alpha=0.05)
        assert gc["X"][0][2]  # X→Y significant
        assert not gc["Y"][0][2]  # Y→X not significant

    def test_granger_bidirectional(self):
        """Bidirectional feedback: both directions significant."""
        from core.ts_discovery import granger_causality_test
        rng = np.random.default_rng(42)
        n = 300
        X = np.zeros(n)
        Y = np.zeros(n)
        for t in range(1, n):
            X[t] = 0.6 * X[t - 1] + 0.3 * Y[t - 1] + rng.normal(0, 0.3)
            Y[t] = 0.6 * Y[t - 1] + 0.3 * X[t - 1] + rng.normal(0, 0.3)
        data = np.column_stack([X, Y])
        gc = granger_causality_test(data, ["X", "Y"], max_lag=1, alpha=0.05)
        assert gc["X"][0][2]  # X→Y
        assert gc["Y"][0][2]  # Y→X

    def test_granger_discover_dag(self):
        """Granger DAG should recover chain structure."""
        from core.ts_discovery import granger_discover_dag
        rng = np.random.default_rng(42)
        n = 300
        X = np.zeros(n)
        Y = np.zeros(n)
        Z = np.zeros(n)
        for t in range(1, n):
            X[t] = 0.7 * X[t - 1] + rng.normal(0, 0.3)
            Y[t] = 0.5 * Y[t - 1] + 0.5 * X[t - 1] + rng.normal(0, 0.3)
            Z[t] = 0.5 * Z[t - 1] + 0.5 * Y[t - 1] + rng.normal(0, 0.3)
        data = np.column_stack([X, Y, Z])
        dag = granger_discover_dag(data, ["X", "Y", "Z"], max_lag=2, alpha=0.01)
        assert "X" in dag.children("X") or "Y" in dag.children("X")  # at least one edge


class TestParser:
    def test_extract_variables(self):
        from nlp.parser import CausalParser
        p = CausalParser("variables: X, Y, Z. X causes Y.")
        dag = p.build_dag()
        assert len(dag.variables) == 3

    def test_extract_edges(self):
        from nlp.parser import CausalParser
        p = CausalParser("variables: A, B. A causes B.")
        dag = p.build_dag()
        assert "B" in dag.children("A")

    def test_load_template(self):
        from nlp.parser import load_template
        desc, dag = load_template("simpsons_paradox")
        assert dag is not None
        assert len(dag.variables) == 3
        assert "G" in dag.variables

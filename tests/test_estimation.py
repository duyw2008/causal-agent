"""Tests for ATE estimators and modern methods."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.graph import CausalDAG
from core.discovery import generate_linear_data
from core.estimation import estimate_effect, CausalEstimate
from core.modern import (
    estimate_ate_dml, estimate_cate_slearner,
    estimate_cate_tlearner, estimate_cate_xlearner,
)


class TestEstimation:
    @classmethod
    def setup_class(cls):
        dag = CausalDAG(["G", "D", "R"], [("G", "D"), ("G", "R"), ("D", "R")])
        cls.data = generate_linear_data(dag, 1000, seed=42)
        cls.vars = ["G", "D", "R"]

    def test_linear_significant(self):
        est = estimate_effect(self.data, self.vars, "D", "R", ["G"], "linear")
        assert est.is_significant()
        assert isinstance(est, CausalEstimate)
        assert est.ci_lower < est.ate < est.ci_upper

    def test_all_methods_return_estimate(self):
        for method in ["linear", "psm", "ipw", "dr", "stratified"]:
            est = estimate_effect(self.data, self.vars, "D", "R", ["G"], method)
            assert isinstance(est.ate, float)
            assert est.std_error > 0

    def test_auto_method(self):
        est = estimate_effect(self.data, self.vars, "D", "R", ["G"], "auto")
        assert est.is_significant()

    def test_dml_significant(self):
        est = estimate_ate_dml(self.data, self.vars, "D", "R", ["G"], n_folds=5)
        assert est.is_significant()
        assert isinstance(est, CausalEstimate)

    def test_cate_slearner(self):
        cate = estimate_cate_slearner(self.data, self.vars, "D", "R", ["G"])
        assert cate.cate.shape[0] == len(self.data)
        assert isinstance(cate.ate, float)

    def test_cate_tlearner(self):
        cate = estimate_cate_tlearner(self.data, self.vars, "D", "R", ["G"])
        assert cate.cate.shape[0] == len(self.data)
        assert isinstance(cate.ate, float)

    def test_cate_xlearner(self):
        cate = estimate_cate_xlearner(self.data, self.vars, "D", "R", ["G"])
        assert cate.cate.shape[0] == len(self.data)
        assert isinstance(cate.ate, float)

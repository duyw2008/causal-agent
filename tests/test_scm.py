"""Tests for SCM, do(), and counterfactual."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.graph import CausalDAG
from core.scm import linear_scm


class TestSCM:
    def test_counterfactual_exact(self):
        """Smoking→Tar→Cancer: counterfactual should match manual calculation."""
        dag = CausalDAG(
            ["G", "S", "T", "C"],
            [("G", "S"), ("G", "C"), ("S", "T"), ("T", "C"), ("S", "C")],
        )
        scm = linear_scm(dag, coefficients={
            "S": {"G": 0.3},
            "T": {"S": 0.8},
            "C": {"T": 0.5, "S": 0.2, "G": 0.4},
        }, noise_std=0.05)

        cf = scm.counterfactual(
            {"G": 2.0, "S": 3.0, "T": 2.5, "C": 4.0},
            {"S": 0.0}, "C",
        )
        # Expected: T'=0.8*0+0.1=0.1, C'=0.5*0.1+0.2*0+0.4*2+1.35=2.20
        assert abs(cf - 2.20) < 0.02, f"Expected 2.20, got {cf:.4f}"

    def test_intervention_sampling(self):
        """do() should produce different distribution than observational."""
        dag = CausalDAG(["X", "Y"], [("X", "Y")])
        scm = linear_scm(dag, {"Y": {"X": 0.8}}, noise_std=0.1)
        intv = scm.intervene({"X": 0.0})
        data_intv = intv.sample(1000, seed=42)
        # Under do(X=0), E[Y] ≈ 0 (noise only)
        assert abs(data_intv["Y"].mean()) < 0.1

    def test_sample_reproducibility(self):
        """Same seed should produce same samples."""
        dag = CausalDAG(["X", "Y"], [("X", "Y")])
        scm = linear_scm(dag, {"Y": {"X": 0.5}}, noise_std=0.2)
        d1 = scm.sample(100, seed=42)
        d2 = scm.sample(100, seed=42)
        assert np.allclose(d1["X"], d2["X"])
        assert np.allclose(d1["Y"], d2["Y"])

    def test_linear_scm_ate_path(self):
        """Chain: total effect = product of path coefficients."""
        dag = CausalDAG(["X", "M", "Y"], [("X", "M"), ("M", "Y")])
        scm = linear_scm(dag, {"M": {"X": 0.6}, "Y": {"M": 0.5}}, noise_std=0.01)
        data = scm.sample(5000, seed=42)
        # E[Y] ≈ 0 (mean-zero noise), but ATE of X→Y via M = 0.6*0.5 = 0.3
        # Correlation-based check
        corr_xy = np.corrcoef(data["X"], data["Y"])[0, 1]
        assert abs(corr_xy - 0.3) < 0.2  # rough check

    def test_nonlinear_counterfactual(self):
        """Non-linear ANM: Y = X² + 2Z + noise."""
        from core.scm import nonlinear_scm
        dag = CausalDAG(["X", "Z", "Y"], [("X", "Y"), ("Z", "Y")])
        scm = nonlinear_scm(dag, {
            "Y": (lambda x, z: x**2 + 2*z, ["X", "Z"]),
        }, noise_std=0.05)

        obs = {"X": 3.0, "Z": 1.0, "Y": 12.0}
        cf = scm.counterfactual(obs, {"X": 0.0}, "Y")
        # noise = 12 - (9+2) = 1; cf = 0 + 2 + 1 = 3
        assert abs(cf - 3.0) < 0.1

    def test_nonlinear_cobb_douglas(self):
        """Cobb-Douglas production: Output = 2·L^0.7·K^0.3 + noise."""
        from core.scm import nonlinear_scm
        dag = CausalDAG(["L", "K", "Q"], [("L", "Q"), ("K", "Q")])
        scm = nonlinear_scm(dag, {
            "Q": (lambda l, k: 2.0 * (l**0.7) * (k**0.3), ["L", "K"]),
        }, noise_std=0.1)

        obs = {"L": 10.0, "K": 5.0, "Q": 12.0}
        cf = scm.counterfactual(obs, {"L": 5.0}, "Q")
        # Should be different from observed
        assert abs(cf - obs["Q"]) > 0.5  # counterfactual ≠ observed

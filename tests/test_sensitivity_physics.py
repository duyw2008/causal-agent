"""Tests for sensitivity analysis and physics engine."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.graph import CausalDAG
from core.discovery import generate_linear_data
from core.estimation import estimate_effect
from core.sensitivity import e_value, rosenbaum_bounds, full_sensitivity_report


class TestSensitivity:
    @classmethod
    def setup_class(cls):
        dag = CausalDAG(["G", "D", "R"], [("G", "D"), ("G", "R"), ("D", "R")])
        data = generate_linear_data(dag, 1000, seed=42)
        cls.est = estimate_effect(data, ["G", "D", "R"], "D", "R", ["G"], "linear")

    def test_evalue_positive(self):
        ev = e_value(self.est.ate, self.est.std_error)
        assert ev.e_value > 1.0

    def test_evalue_ci_positive(self):
        ev = e_value(self.est.ate, self.est.std_error)
        assert ev.e_value_ci is not None

    def test_rosenbaum_threshold(self):
        rb = rosenbaum_bounds(self.est.ate, self.est.std_error)
        assert rb.gamma_threshold >= 1.0

    def test_full_report(self):
        report = full_sensitivity_report(self.est.ate, self.est.std_error)
        assert "SENSITIVITY" in report
        assert "E-value" in report
        assert "Rosenbaum" in report


class TestPhysics:
    def test_library_has_laws(self):
        from core.physics import PhysicsLibrary
        lib = PhysicsLibrary()
        assert len(lib.list_all()) >= 14

    def test_domains(self):
        from core.physics import PhysicsLibrary
        lib = PhysicsLibrary()
        assert "mechanics" in lib.domains()
        assert "electromagnetism" in lib.domains()

    def test_find_relevant(self):
        from core.physics import PhysicsLibrary
        lib = PhysicsLibrary()
        laws = lib.find_relevant(["Force", "Mass", "Acceleration"])
        assert len(laws) >= 2  # newton_2nd, kinetic_energy at minimum

    def test_pendulum_pipeline(self):
        from core.physics import physics_causal_pipeline
        rng = np.random.default_rng(42)
        L = rng.uniform(0.5, 2.0, 300)
        g = np.full(300, 9.81) + rng.normal(0, 0.01, 300)
        T = 2 * np.pi * np.sqrt(L / g) + rng.normal(0, 0.02, 300)
        data = np.column_stack([L, g, T])
        result = physics_causal_pipeline(
            data, ["Length", "Gravity", "Period"], "Length", "Period"
        )
        assert "Period" in result["physics_equations"]
        assert result["identification"]["identifiable"]

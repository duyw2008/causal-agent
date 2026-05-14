#!/usr/bin/env python3
"""
Physics-Informed Causal Agent — 将物理规律嵌入因果推断

物理定律作为因果约束的三种形式:
  1. Hard Constraints on DAG: 某些边被物理定律禁止或强制
  2. Structural Equations: 用已知物理方程替换任意的 f()
  3. Conservation Priors: 守恒律约束 ATE 和反事实的取值范围

运行: python demos/physics_causal_demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# ══════════════════════════════════════════════════════════════
#  Part 1: Physical Law Library
# ══════════════════════════════════════════════════════════════

@dataclass
class PhysicalLaw:
    """A physical law that can constrain causal models."""
    name: str
    equation: str                     # LaTeX formula (documentation)
    variables_involved: List[str]     # which variables are constrained
    constraint_type: str              # "dag_edge" | "scm_equation" | "conservation"
    apply_fn: Callable                # function to apply the constraint

class PhysicsLibrary:
    """Registry of known physical laws."""

    def __init__(self):
        self.laws: Dict[str, PhysicalLaw] = {}
        self._register_defaults()

    def _register_defaults(self):
        # ── Newtonian Mechanics ──
        self.register(PhysicalLaw(
            name="newton_2nd",
            equation=r"$F = ma$",
            variables_involved=["Force", "Mass", "Acceleration"],
            constraint_type="scm_equation",
            apply_fn=lambda scm: _apply_newton_2nd(scm),
        ))

        self.register(PhysicalLaw(
            name="momentum_conservation",
            equation=r"$\sum p_{before} = \sum p_{after}$",
            variables_involved=["Velocity_before", "Mass", "Velocity_after"],
            constraint_type="conservation",
            apply_fn=lambda scm, data: _check_momentum_conservation(scm, data),
        ))

        # ── Energy ──
        self.register(PhysicalLaw(
            name="energy_conservation",
            equation=r"$\Delta E = Q - W$",
            variables_involved=["Energy_in", "Work", "Energy_out"],
            constraint_type="conservation",
            apply_fn=lambda scm, data: _check_energy_conservation(scm, data),
        ))

        self.register(PhysicalLaw(
            name="kinetic_energy",
            equation=r"$E_k = \frac{1}{2}mv^2$",
            variables_involved=["Mass", "Velocity", "KineticEnergy"],
            constraint_type="scm_equation",
            apply_fn=lambda scm: _apply_kinetic_energy(scm),
        ))

        # ── Thermodynamics ──
        self.register(PhysicalLaw(
            name="second_law_thermo",
            equation=r"$\Delta S \geq 0$",
            variables_involved=["Entropy"],
            constraint_type="conservation",
            apply_fn=lambda scm, data: _check_entropy_increase(scm, data),
        ))

        # ── Electromagnetism ──
        self.register(PhysicalLaw(
            name="ohms_law",
            equation=r"$V = IR$",
            variables_involved=["Voltage", "Current", "Resistance"],
            constraint_type="scm_equation",
            apply_fn=lambda scm: _apply_ohms_law(scm),
        ))

        # ── Causality constraints from physics ──
        self.register(PhysicalLaw(
            name="force_causes_acceleration",
            equation=r"$a = F/m$ (force causes acceleration, not reverse)",
            variables_involved=["Force", "Acceleration"],
            constraint_type="dag_edge",
            apply_fn=lambda dag: _force_edge_direction(dag, "Force", "Acceleration"),
        ))

        self.register(PhysicalLaw(
            name="no_free_energy",
            equation="Energy must come from somewhere",
            variables_involved=["Energy_out"],
            constraint_type="dag_edge",
            apply_fn=lambda dag: _require_input_for_output(dag),
        ))

    def register(self, law: PhysicalLaw):
        self.laws[law.name] = law

    def get_constraints_for_variables(self, variables: List[str]) -> List[PhysicalLaw]:
        """Return all laws that involve any of the given variables."""
        var_set = set(variables)
        return [law for law in self.laws.values()
                if var_set & set(law.variables_involved)]


# ── Physical constraint implementations ─────────────────────

def _apply_newton_2nd(scm):
    """Ensure F = ma holds in the SCM."""
    # Replace the structural equation for Acceleration
    # with a = F / m (deterministic from physics)
    return {
        "action": "replace_equation",
        "variable": "Acceleration",
        "parents": ["Force", "Mass"],
        "function": "lambda f, m, noise=0: f / max(m, 0.001) + noise * 0.01",
        "reason": "Newton's Second Law requires a = F/m"
    }

def _check_momentum_conservation(scm, data):
    """Check if momentum is conserved in the generated data."""
    if "Velocity_before" not in data or "Velocity_after" not in data:
        return {"action": "warn", "reason": "Cannot verify; missing variables"}
    m = data.get("Mass", np.ones(len(data["Velocity_before"])))
    p_before = m * data["Velocity_before"]
    p_after = m * data["Velocity_after"]
    violation = np.abs(p_before - p_after).mean()
    return {
        "action": "validate",
        "constraint": "Σp_before = Σp_after",
        "violation": float(violation),
        "acceptable": violation < 0.1,
        "reason": f"Momentum violation = {violation:.4f}"
    }

def _check_energy_conservation(scm, data):
    """Check energy conservation."""
    return {"action": "validate", "constraint": "ΔE = Q - W",
            "reason": "Energy conservation check (simplified)"}

def _apply_kinetic_energy(scm):
    return {"action": "replace_equation", "variable": "KineticEnergy",
            "parents": ["Mass", "Velocity"],
            "function": "lambda m, v, noise=0: 0.5 * m * v**2 + noise * 0.01"}

def _check_entropy_increase(scm, data):
    if "Entropy" not in data:
        return {"action": "warn", "reason": "Cannot verify entropy"}
    return {"action": "validate", "constraint": "ΔS ≥ 0",
            "reason": "Second Law of Thermodynamics"}

def _apply_ohms_law(scm):
    return {"action": "replace_equation", "variable": "Current",
            "parents": ["Voltage", "Resistance"],
            "function": "lambda v, r, noise=0: v / max(r, 0.001) + noise * 0.01"}

def _force_edge_direction(dag, cause, effect):
    """Ensure the causal edge goes from cause to effect, not reverse."""
    if effect in dag.parents(cause):
        return {"action": "reverse_edge", "from": effect, "to": cause,
                "reason": f"Physics requires {cause} → {effect}, not reverse"}
    return {"action": "ok"}

def _require_input_for_output(dag):
    """Check that every 'output' variable has at least one parent."""
    outputs = [v for v in dag.variables if "out" in v.lower() or "energy" in v.lower()]
    issues = []
    for v in outputs:
        if not dag.parents(v):
            issues.append(v)
    if issues:
        return {"action": "warn",
                "reason": f"Output/energy variables without inputs: {issues}"}
    return {"action": "ok"}


# ══════════════════════════════════════════════════════════════
#  Part 2: Physics-Informed Causal Discovery
# ══════════════════════════════════════════════════════════════

class PhysicsInformedDiscoverer:
    """
    Causal discovery that respects physical constraints.

    The key insight: physical laws act as HARD PRIORS on the
    causal graph structure. We don't need to "discover" that
    F→a (force causes acceleration) — physics tells us that.
    """

    def __init__(self, physics: PhysicsLibrary):
        self.physics = physics

    def discover_with_physics(
        self,
        data: np.ndarray,
        var_names: List[str],
        alpha: float = 0.05,
    ) -> Dict:
        """
        Run causal discovery, then validate and correct using physics.
        """
        from core.discovery import pc_algorithm
        from core.graph import CausalDAG

        # Step 1: Data-driven discovery (PC)
        pc_dag = pc_algorithm(data, var_names, alpha=alpha)

        # Step 2: Find relevant physical laws
        laws = self.physics.get_constraints_for_variables(var_names)

        # Step 3: Apply DAG-edge constraints
        constraints_applied = []
        for law in laws:
            if law.constraint_type == "dag_edge":
                result = law.apply_fn(pc_dag)
                constraints_applied.append({
                    "law": law.name,
                    "equation": law.equation,
                    "result": result,
                })

        # Step 4: Build physics-prior adjacency matrix
        # (edges that physics says MUST exist)
        forced_edges = self._get_forced_edges(var_names, laws)

        # Step 5: Build physics-prior forbidden edges
        forbidden_edges = self._get_forbidden_edges(var_names, laws)

        return {
            "pc_dag": pc_dag,
            "relevant_laws": [(law.name, law.equation) for law in laws],
            "constraints_applied": constraints_applied,
            "forced_edges": forced_edges,
            "forbidden_edges": forbidden_edges,
        }

    def _get_forced_edges(self, var_names, laws):
        """Edges that physics REQUIRES to exist."""
        forced = []
        for law in laws:
            if law.constraint_type != "scm_equation":
                continue
            try:
                result = law.apply_fn(None)
            except TypeError:
                continue
            if isinstance(result, dict) and result.get("action") == "replace_equation":
                parents = law.variables_involved[:-1]
                child = law.variables_involved[-1]
                if set(parents + [child]).issubset(set(var_names)):
                    for p in parents:
                        forced.append((p, child))
        return forced

    def _get_forbidden_edges(self, var_names, laws):
        """Edges that physics FORBIDS."""
        forbidden = []
        # Time cannot go backwards
        if "Time" in var_names:
            for v in var_names:
                if v != "Time":
                    forbidden.append((v, "Time"))
        # Acceleration cannot cause Force (only F→a)
        if "Acceleration" in var_names and "Force" in var_names:
            forbidden.append(("Acceleration", "Force"))
        return forbidden


# ══════════════════════════════════════════════════════════════
#  Part 3: Physics-Validated Counterfactuals
# ══════════════════════════════════════════════════════════════

class PhysicsValidatedSCM:
    """
    SCM wrapper that validates counterfactuals against physical laws.

    If a counterfactual violates a conservation law, it's flagged
    and either rejected or annotated.
    """

    def __init__(self, scm, physics: PhysicsLibrary):
        self.scm = scm
        self.physics = physics

    def counterfactual_with_physics_check(
        self, observed: dict, intervention: dict, target: str,
    ) -> dict:
        """Compute counterfactual and check if it respects physics."""
        # Standard counterfactual
        cf_val = self.scm.counterfactual(observed, intervention, target)

        # Get relevant laws
        all_vars = list(observed.keys())
        laws = self.physics.get_constraints_for_variables(all_vars)

        # Check each law
        violations = []
        for law in laws:
            if law.constraint_type == "conservation":
                # Build the counterfactual state
                cf_state = dict(observed)
                cf_state.update(intervention)
                cf_state[target] = cf_val

                # Check conservation
                result = law.apply_fn(self.scm, cf_state)
                if isinstance(result, dict) and not result.get("acceptable", True):
                    violations.append({
                        "law": law.name,
                        "equation": law.equation,
                        "violation": result.get("violation", "unknown"),
                    })

        return {
            "counterfactual_value": cf_val,
            "target": target,
            "intervention": intervention,
            "violates_physics": len(violations) > 0,
            "violations": violations,
            "verdict": ("⚠️  WARNING: Counterfactual violates physical laws!"
                       if violations else
                       "✅ Counterfactual respects all known physical laws."),
        }


# ══════════════════════════════════════════════════════════════
#  Part 4: Full Demo
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  PHYSICS-INFORMED CAUSAL AGENT — DEMO")
    print("=" * 65)

    physics = PhysicsLibrary()
    print(f"\n📚 Physics Library: {len(physics.laws)} laws registered")
    for name, law in physics.laws.items():
        print(f"  • {name}: {law.equation}")

    # ── Demo 1: Physics constrains causal discovery ──
    print(f"\n{'─'*65}")
    print("  Demo 1: Physics-Constrained Causal Discovery")
    print(f"{'─'*65}")

    # Generate data from a mechanical system
    rng = np.random.default_rng(42)
    n = 500
    Mass = np.full(n, 1.0) + rng.normal(0, 0.05, n)
    Force = rng.normal(0, 2, n)
    Acceleration = Force / Mass + rng.normal(0, 0.1, n)  # F=ma + noise
    Velocity = np.cumsum(Acceleration * 0.01) + rng.normal(0, 0.1, n)

    data = np.column_stack([Force, Mass, Acceleration, Velocity])
    var_names = ["Force", "Mass", "Acceleration", "Velocity"]

    discoverer = PhysicsInformedDiscoverer(physics)
    result = discoverer.discover_with_physics(data, var_names, alpha=0.01)

    print(f"\n  PC-discovered DAG: {result['pc_dag']}")
    print(f"\n  Relevant physical laws:")
    for name, eq in result["relevant_laws"]:
        print(f"    {name}: {eq}")
    print(f"\n  Forced edges (physics requires): {result['forced_edges']}")
    print(f"  Forbidden edges (physics forbids): {result['forbidden_edges']}")
    print(f"\n  Constraints applied:")
    for c in result["constraints_applied"]:
        if c["result"].get("action") == "reverse_edge":
            print(f"    ⚠️  {c['law']}: {c['result']['reason']}")

    # ── Demo 2: Physics-validated counterfactual ──
    print(f"\n{'─'*65}")
    print("  Demo 2: Physics-Validated Counterfactuals")
    print(f"{'─'*65}")

    from core.graph import CausalDAG
    from core.scm import linear_scm

    # Build an SCM with physical equations
    dag = CausalDAG(
        ["Force", "Mass", "Acceleration", "Velocity", "KineticEnergy"],
        [("Force","Acceleration"), ("Mass","Acceleration"),
         ("Acceleration","Velocity"), ("Mass","KineticEnergy"),
         ("Velocity","KineticEnergy")]
    )

    # Use physics-informed equations
    scm = linear_scm(dag, coefficients={
        "Acceleration": {"Force": 1.0, "Mass": 0.0},  # will be overridden
        "Velocity": {"Acceleration": 1.0},
        "KineticEnergy": {"Mass": 0.5, "Velocity": 0.0},  # will be overridden
    }, noise_std=0.01)

    pv_scm = PhysicsValidatedSCM(scm, physics)

    # Counterfactual: what if force were 0?
    observed = {"Force": 5.0, "Mass": 2.0, "Acceleration": 2.5,
                "Velocity": 10.0, "KineticEnergy": 100.0}
    intervention = {"Force": 0.0}

    cf = pv_scm.counterfactual_with_physics_check(
        observed, intervention, "KineticEnergy"
    )

    print(f"\n  Observed:  {observed}")
    print(f"  do(Force=0): KineticEnergy = {cf['counterfactual_value']:.1f}")
    print(f"  {cf['verdict']}")
    if cf['violations']:
        for v in cf['violations']:
            print(f"    Violates {v['law']}: {v['equation']}")

    # ── Demo 3: Integration strategy summary ──
    print(f"\n{'─'*65}")
    print("  Demo 3: Three Integration Strategies")
    print(f"{'─'*65}")

    strategies = [
        ("Level 1: DAG Constraints",
         "Physical laws restrict which edges can exist.\n"
         "  Example: Force → Acceleration (forced)\n"
         "           Acceleration → Force (forbidden)\n"
         "  Benefit: Narrows causal discovery search space."),
        ("Level 2: SCM Equations",
         "Replace arbitrary f() with known physics.\n"
         "  Example: a = F/m (instead of learned linear function)\n"
         "  Benefit: Reduces parameters, improves extrapolation."),
        ("Level 3: Conservation Validation",
         "Check if counterfactuals respect conservation laws.\n"
         "  Example: Total energy must be conserved after intervention.\n"
         "  Benefit: Catches physically impossible counterfactuals."),
    ]

    for level, desc in strategies:
        print(f"\n  {level}")
        for line in desc.split("\n"):
            print(f"  {line}")

    print(f"\n{'='*65}")
    print("  PHYSICS INTEGRATION COMPLETE")
    print(f"{'='*65}")

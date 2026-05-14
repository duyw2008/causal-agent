"""
Physics-Informed Causal Engine — 物理规律作为因果推断的第一性约束

Architecture:
  PhysicsLaw       — 抽象物理定律（符号公式 + 因果约束）
  PhysicsLibrary   — 按领域组织的物理定律库
  PhysicsInformedCausalGraph  — 物理约束的因果 DAG
  PhysicsInformedSCM          — 物理约束的结构因果模型
  SymbolicPhysicsDiscovery    — 从数据中发现符号化物理公式

Design principle:
  因果推断回答"谁影响谁" → 物理定律回答"影响必须是怎样的"
  两者结合: 因果结构提供骨架, 物理定律填充血肉。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from .graph import CausalDAG
from .scm import SCM, StructuralEquation, linear_scm


# ═══════════════════════════════════════════════════════════════════
#  Constraint Type
# ═══════════════════════════════════════════════════════════════════

class ConstraintType(Enum):
    DAG_EDGE = "dag_edge"           # 强制/禁止因果边
    SCM_EQUATION = "scm_equation"   # 替换结构方程
    CONSERVATION = "conservation"    # 守恒律验证
    BOUNDARY = "boundary"           # 取值范围约束
    SYMMETRY = "symmetry"           # 对称性约束


# ═══════════════════════════════════════════════════════════════════
#  Physics Law
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PhysicsLaw:
    """An immutable physical law that constrains causal models."""

    name: str                          # 唯一标识
    domain: str                        # 领域: mechanics, thermo, em, ...
    latex: str                         # LaTeX 表达式

    # ── variables ──
    inputs: List[str]                  # 输入/原因变量
    outputs: List[str]                 # 输出/结果变量
    parameters: List[str] = field(default_factory=list)  # 常数参数

    # ── constraints ──
    constraint_type: ConstraintType = ConstraintType.SCM_EQUATION

    # ── formula ──
    formula: Optional[str] = None      # Python eval-able formula string
    # e.g. "lambda f,m: f / m"  for a = F/m

    # ── causal direction ──
    causal_direction: List[Tuple[str, str]] = field(default_factory=list)
    # [(cause, effect), ...] — 物理强制因果方向

    forbidden_directions: List[Tuple[str, str]] = field(default_factory=list)
    # 物理禁止的因果方向

    required_parents: Dict[str, List[str]] = field(default_factory=dict)
    # {variable: [must_be_parents]} — 守恒律要求

    # ── validation ──
    tolerance: float = 0.05            # 验证容差

    def apply_to_dag(self, dag: CausalDAG) -> Dict:
        """Apply DAG-level constraints. Returns list of actions."""
        actions = []

        # Enforce causal directions
        for cause, effect in self.causal_direction:
            if effect in dag.parents(cause):
                actions.append({
                    "action": "reverse_edge",
                    "from": effect, "to": cause,
                    "law": self.name,
                    "reason": f"{self.latex}: {cause} must cause {effect}"
                })
            elif cause not in dag.parents(effect) and cause in dag.variables and effect in dag.variables:
                actions.append({
                    "action": "suggest_edge",
                    "from": cause, "to": effect,
                    "law": self.name,
                    "reason": f"{self.latex}: {cause} should cause {effect}"
                })

        # Forbidden directions
        for cause, effect in self.forbidden_directions:
            if cause in dag.children(effect):
                actions.append({
                    "action": "forbid_edge",
                    "from": effect, "to": cause,
                    "law": self.name,
                    "reason": f"{self.latex}: {effect} → {cause} is physically impossible"
                })

        # Required parents
        for var, required in self.required_parents.items():
            if var in dag.variables:
                current = dag.parents(var)
                missing = [p for p in required if p not in current and p in dag.variables]
                if missing:
                    actions.append({
                        "action": "require_parents",
                        "variable": var,
                        "missing": missing,
                        "law": self.name,
                        "reason": f"{self.latex}: {var} requires inputs {missing}"
                    })

        return {"actions": actions, "law": self.name}

    def apply_to_scm(self, scm: SCM) -> Dict:
        """Replace structural equations with physics formulas."""
        if self.constraint_type != ConstraintType.SCM_EQUATION or not self.formula:
            return {"action": "none", "reason": "No SCM equation defined"}

        replacements = []
        for out_var in self.outputs:
            if out_var in scm.variables:
                eq = StructuralEquation(
                    variable=out_var,
                    func=eval(self.formula),
                    parents=self.inputs,
                )
                replacements.append({
                    "variable": out_var,
                    "old_parents": list(scm.equations[out_var].parents),
                    "new_parents": self.inputs,
                    "equation": self.latex,
                })

        return {
            "action": "replace_equations",
            "replacements": replacements,
            "law": self.name,
        }

    def validate(self, data: Dict[str, np.ndarray]) -> Dict:
        """Check if data obeys this physical law."""
        if self.constraint_type != ConstraintType.CONSERVATION:
            return {"action": "skip"}

        missing = [v for v in self.inputs + self.outputs if v not in data]
        if missing:
            return {"action": "skip", "reason": f"Missing variables: {missing}"}

        # Evaluate conservation check
        if self.formula:
            try:
                violation = eval(self.formula, {"np": np}, data)
                acceptable = abs(violation) < self.tolerance if isinstance(violation, (int, float)) else np.all(np.abs(violation) < self.tolerance)
                return {
                    "action": "validate",
                    "acceptable": bool(acceptable),
                    "violation": float(np.mean(np.abs(violation))) if hasattr(violation, '__len__') else float(violation),
                    "law": self.name,
                    "equation": self.latex,
                }
            except Exception as e:
                return {"action": "error", "reason": str(e)}

        return {"action": "skip", "reason": "No validation formula"}

    def summary(self) -> str:
        return f"[{self.domain}] {self.name}: {self.latex}"


# ═══════════════════════════════════════════════════════════════════
#  Physics Library
# ═══════════════════════════════════════════════════════════════════

class PhysicsLibrary:
    """Curated library of physical laws, organized by domain."""

    def __init__(self):
        self._laws: Dict[str, PhysicsLaw] = {}
        self._by_domain: Dict[str, List[str]] = {}
        self._register_core_laws()

    def _register_core_laws(self):
        """Register fundamental physical laws."""
        core = [

            # ═══ Mechanics ═══
            PhysicsLaw(
                name="newton_2nd",
                domain="mechanics",
                latex=r"$F = ma$",
                inputs=["Force", "Mass"],
                outputs=["Acceleration"],
                constraint_type=ConstraintType.SCM_EQUATION,
                formula="lambda f,m: f / max(m, 0.001)",
                causal_direction=[("Force", "Acceleration")],
                forbidden_directions=[("Acceleration", "Force")],
            ),
            PhysicsLaw(
                name="kinetic_energy",
                domain="mechanics",
                latex=r"$E_k = \frac{1}{2}mv^2$",
                inputs=["Mass", "Velocity"],
                outputs=["KineticEnergy"],
                constraint_type=ConstraintType.SCM_EQUATION,
                formula="lambda m,v: 0.5 * m * v**2",
                causal_direction=[("Mass","KineticEnergy"), ("Velocity","KineticEnergy")],
            ),
            PhysicsLaw(
                name="momentum",
                domain="mechanics",
                latex=r"$p = mv$",
                inputs=["Mass", "Velocity"],
                outputs=["Momentum"],
                constraint_type=ConstraintType.SCM_EQUATION,
                formula="lambda m,v: m * v",
                causal_direction=[("Mass","Momentum"), ("Velocity","Momentum")],
            ),
            PhysicsLaw(
                name="momentum_conservation",
                domain="mechanics",
                latex=r"$\sum p_{before} = \sum p_{after}$",
                inputs=["Momentum_before", "Momentum_after"],
                outputs=[],
                constraint_type=ConstraintType.CONSERVATION,
                formula="np.mean(np.abs(data['Momentum_before'] - data['Momentum_after']))",
                required_parents={},
            ),
            PhysicsLaw(
                name="energy_conservation",
                domain="mechanics",
                latex=r"$\Delta E_{total} = 0$",
                inputs=["Energy_in", "Energy_out"],
                outputs=[],
                constraint_type=ConstraintType.CONSERVATION,
                formula="np.mean(np.abs(data['Energy_in'] - data['Energy_out']))",
            ),
            PhysicsLaw(
                name="hookes_law",
                domain="mechanics",
                latex=r"$F = -kx$",
                inputs=["Displacement", "SpringConstant"],
                outputs=["RestoringForce"],
                constraint_type=ConstraintType.SCM_EQUATION,
                formula="lambda x,k: -k * x",
                causal_direction=[("Displacement","RestoringForce")],
            ),
            PhysicsLaw(
                name="pendulum_period",
                domain="mechanics",
                latex=r"$T = 2\pi\sqrt{L/g}$",
                inputs=["Length"],
                outputs=["Period"],
                parameters=["Gravity"],
                constraint_type=ConstraintType.SCM_EQUATION,
                formula="lambda L, g=9.81: 2 * 3.14159 * (L / g)**0.5",
                causal_direction=[("Length","Period")],
            ),

            # ═══ Electromagnetism ═══
            PhysicsLaw(
                name="ohms_law",
                domain="electromagnetism",
                latex=r"$V = IR$",
                inputs=["Voltage", "Resistance"],
                outputs=["Current"],
                constraint_type=ConstraintType.SCM_EQUATION,
                formula="lambda v,r: v / max(r, 0.001)",
                causal_direction=[("Voltage","Current")],
            ),
            PhysicsLaw(
                name="power_law",
                domain="electromagnetism",
                latex=r"$P = IV$",
                inputs=["Current", "Voltage"],
                outputs=["Power"],
                constraint_type=ConstraintType.SCM_EQUATION,
                formula="lambda i,v: i * v",
                causal_direction=[("Current","Power"), ("Voltage","Power")],
                required_parents={"Power": ["Current", "Voltage"]},
            ),
            PhysicsLaw(
                name="joules_law",
                domain="electromagnetism",
                latex=r"$Q = I^2Rt$",
                inputs=["Current", "Resistance", "Time"],
                outputs=["Heat"],
                constraint_type=ConstraintType.SCM_EQUATION,
                formula="lambda i,r,t: i**2 * r * t",
                causal_direction=[("Current","Heat"), ("Resistance","Heat"), ("Time","Heat")],
            ),

            # ═══ Thermodynamics ═══
            PhysicsLaw(
                name="ideal_gas_law",
                domain="thermodynamics",
                latex=r"$PV = nRT$",
                inputs=["Pressure", "Volume"],
                outputs=["Temperature"],
                parameters=["n", "R"],
                constraint_type=ConstraintType.SCM_EQUATION,
                formula="lambda p,v, n=1.0, R=8.314: (p * v) / (n * max(R, 0.001))",
                causal_direction=[("Pressure","Temperature"), ("Volume","Temperature")],
            ),
            PhysicsLaw(
                name="second_law_thermo",
                domain="thermodynamics",
                latex=r"$\Delta S \geq 0$",
                inputs=["Entropy_initial", "Entropy_final"],
                outputs=[],
                constraint_type=ConstraintType.CONSERVATION,
                formula="np.mean(np.maximum(0, data['Entropy_initial'] - data['Entropy_final']))",
            ),

            # ═══ Fluids ═══
            PhysicsLaw(
                name="bernoulli",
                domain="fluids",
                latex=r"$P + \frac{1}{2}\rho v^2 + \rho gh = const$",
                inputs=["Pressure", "Velocity", "Height"],
                outputs=[],
                parameters=["Density", "Gravity"],
                constraint_type=ConstraintType.CONSERVATION,
            ),
            PhysicsLaw(
                name="continuity",
                domain="fluids",
                latex=r"$A_1v_1 = A_2v_2$",
                inputs=["Area_in", "Velocity_in", "Area_out"],
                outputs=["Velocity_out"],
                constraint_type=ConstraintType.SCM_EQUATION,
                formula="lambda a1,v1,a2: (a1 * v1) / max(a2, 0.001)",
                causal_direction=[("Area_in","Velocity_out"), ("Velocity_in","Velocity_out"), ("Area_out","Velocity_out")],
            ),
        ]

        for law in core:
            self.register(law)

    def register(self, law: PhysicsLaw):
        self._laws[law.name] = law
        self._by_domain.setdefault(law.domain, []).append(law.name)

    def get(self, name: str) -> Optional[PhysicsLaw]:
        return self._laws.get(name)

    def list_by_domain(self, domain: str) -> List[PhysicsLaw]:
        return [self._laws[n] for n in self._by_domain.get(domain, [])]

    def list_all(self) -> List[PhysicsLaw]:
        return list(self._laws.values())

    def find_relevant(self, variables: List[str]) -> List[PhysicsLaw]:
        """Find all laws that involve any of the given variables."""
        var_set = set(variables)
        return [law for law in self._laws.values()
                if var_set & set(law.inputs + law.outputs)]

    def domains(self) -> List[str]:
        return list(self._by_domain.keys())

    def summary(self) -> str:
        lines = [f"Physics Library: {len(self._laws)} laws in {len(self._by_domain)} domains"]
        for domain in sorted(self._by_domain.keys()):
            laws = self._by_domain[domain]
            lines.append(f"  {domain} ({len(laws)}): {', '.join(laws)}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Physics-Informed Causal Graph
# ═══════════════════════════════════════════════════════════════════

class PhysicsInformedCausalGraph:
    """CausalDAG augmented with physics-derived constraints."""

    def __init__(self, dag: CausalDAG, physics: PhysicsLibrary):
        self.dag = dag
        self.physics = physics
        self._constraints: Dict[str, Dict] = {}
        self._apply_all()

    def _apply_all(self):
        """Apply all relevant physical laws as DAG constraints."""
        relevant = self.physics.find_relevant(self.dag.variables)
        for law in relevant:
            result = law.apply_to_dag(self.dag)
            self._constraints[law.name] = result

    @property
    def forced_edges(self) -> List[Tuple[str, str]]:
        """Edges that physics REQUIRES."""
        edges = []
        for result in self._constraints.values():
            for a in result.get("actions", []):
                if a["action"] == "suggest_edge":
                    edges.append((a["from"], a["to"]))
        return edges

    @property
    def forbidden_edges(self) -> List[Tuple[str, str]]:
        """Edges that physics FORBIDS."""
        edges = []
        for result in self._constraints.values():
            for a in result.get("actions", []):
                if a["action"] == "forbid_edge":
                    edges.append((a["from"], a["to"]))
        return edges

    @property
    def reversed_edges(self) -> List[Tuple[str, str]]:
        """Edges that need reversal per physics."""
        edges = []
        for result in self._constraints.values():
            for a in result.get("actions", []):
                if a["action"] == "reverse_edge":
                    edges.append((a["from"], a["to"]))
        return edges

    def to_physics_corrected_dag(self) -> CausalDAG:
        """Return a new DAG with physics constraints applied."""
        new_edges = set()
        for v in self.dag.variables:
            for c in self.dag.children(v):
                new_edges.add((v, c))

        # Apply reversals
        for old_from, old_to in self.reversed_edges:
            new_edges.discard((old_from, old_to))
            new_edges.add((old_to, old_from))

        return CausalDAG(self.dag.variables, list(new_edges))

    def constraint_report(self) -> str:
        """Human-readable report of all physics constraints."""
        lines = ["Physics Constraints on Causal Graph:", ""]
        for law_name, result in self._constraints.items():
            law = self.physics.get(law_name)
            actions = result.get("actions", [])
            if not actions:
                continue
            lines.append(f"  {law_name} ({law.latex}):")
            for a in actions:
                lines.append(f"    [{a['action']}] {a.get('reason', '')}")
        return "\n".join(lines) if len(lines) > 2 else "No physics constraints apply."


# ═══════════════════════════════════════════════════════════════════
#  Physics-Informed SCM
# ═══════════════════════════════════════════════════════════════════

class PhysicsInformedSCM:
    """SCM where structural equations are replaced by physical laws."""

    def __init__(self, scm: SCM, physics: PhysicsLibrary):
        self.scm = scm
        self.physics = physics
        self._replacements: Dict[str, Dict] = {}
        self._apply_equations()

    def _apply_equations(self):
        relevant = self.physics.find_relevant(self.scm.variables)
        for law in relevant:
            if law.constraint_type == ConstraintType.SCM_EQUATION:
                result = law.apply_to_scm(self.scm)
                if result["action"] == "replace_equations":
                    self._replacements[law.name] = result

    @property
    def physics_equations(self) -> Dict[str, str]:
        """Map variable → physical equation (LaTeX)."""
        eqs = {}
        for result in self._replacements.values():
            for r in result["replacements"]:
                eqs[r["variable"]] = result["law"]
        return eqs

    def sample(self, n: int = 1, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Sample from the physics-informed SCM.

        Variables governed by physical laws use the physics equation.
        Other variables use the original SCM equations.
        """
        if seed is not None:
            np.random.seed(seed)

        order = self.scm.dag.topological_order()
        data: Dict[str, np.ndarray] = {}

        # Determine which variables have physics replacements
        physics_vars = set()
        physics_funcs = {}
        for result in self._replacements.values():
            for r in result["replacements"]:
                physics_vars.add(r["variable"])
                law = self.physics.get(result["law"])
                if law and law.formula:
                    physics_funcs[r["variable"]] = eval(law.formula)

        for v in order:
            if v in physics_vars:
                # Use physics equation
                fn = physics_funcs[v]
                parent_data = [data[p] for p in self.scm.equations[v].parents
                               if p in data]
                if parent_data:
                    data[v] = np.array([fn(*[pd[i] for pd in parent_data])
                                        for i in range(n)])
                else:
                    data[v] = np.full(n, fn())
            else:
                # Use original SCM equation
                eq = self.scm.equations[v]
                noises = np.array([eq.noise_dist() for _ in range(n)])
                if not eq.parents:
                    data[v] = np.array([eq.func(noise=noises[i])
                                        for i in range(n)])
                else:
                    data[v] = np.array([
                        eq.func(
                            *[data[p][i] for p in eq.parents],
                            noise=noises[i],
                        )
                        for i in range(n)
                    ])

        return data

    def counterfactual_with_physics(
        self, observed: Dict[str, float], intervention: Dict[str, Any],
        target: str,
    ) -> Dict:
        """Counterfactual that also checks physics validity."""
        cf_val = self.scm.counterfactual(observed, intervention, target)

        # Build counterfactual state
        cf_state = dict(observed)
        cf_state.update(intervention)
        cf_state[target] = cf_val

        # Validate against conservation laws
        validations = []
        for law in self.physics.list_all():
            if law.constraint_type == ConstraintType.CONSERVATION:
                # Convert to numpy arrays for validation
                arr_data = {k: np.array([v]) for k, v in cf_state.items()}
                result = law.validate(arr_data)
                if result.get("action") == "validate":
                    validations.append({
                        "law": law.name,
                        "equation": law.latex,
                        "acceptable": result["acceptable"],
                        "violation": result.get("violation", 0),
                    })

        violations = [v for v in validations if not v["acceptable"]]

        return {
            "counterfactual_value": cf_val,
            "target": target,
            "intervention": intervention,
            "violates_physics": len(violations) > 0,
            "violations": violations,
            "all_validations": validations,
            "verdict": (
                "⚠️  VIOLATES PHYSICAL LAWS" if violations
                else "✅ Respects all physical laws"
            ),
        }

    def summary(self) -> str:
        lines = ["Physics-Informed SCM:", ""]
        lines.append(f"  Variables: {', '.join(self.scm.variables)}")
        if self.physics_equations:
            lines.append(f"  Physics-governed variables:")
            for var, law_name in self.physics_equations.items():
                law = self.physics.get(law_name)
                lines.append(f"    {var} ← {law.latex}")
        else:
            lines.append(f"  No physics equations applied.")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Symbolic Physics Discovery
# ═══════════════════════════════════════════════════════════════════

class SymbolicPhysicsDiscovery:
    """
    Discover candidate physical formulas from data + causal DAG.

    Searches a space of symbolic expressions (polynomial, trigonometric,
    exponential) and selects the one that best fits the data while
    minimizing complexity (Occam's razor).
    """

    def __init__(self, physics: PhysicsLibrary):
        self.physics = physics

    def discover_formula(
        self,
        data: np.ndarray,
        var_names: List[str],
        target: str,
        candidate_parents: List[str],
        max_degree: int = 2,
        max_terms: int = 4,
    ) -> Dict:
        """
        Search for the best symbolic formula relating target to its parents.

        Returns the discovered formula, its LaTeX rendering, and fit metrics.
        """
        idx = {v: i for i, v in enumerate(var_names)}
        y = data[:, idx[target]]
        parents_data = {p: data[:, idx[p]] for p in candidate_parents}

        # Generate candidate expressions
        candidates = self._generate_expressions(
            candidate_parents, max_degree, max_terms
        )

        # Score each candidate
        best = None
        best_score = -np.inf

        for expr, expr_latex, n_params in candidates:
            try:
                y_pred = self._evaluate_expression(expr, parents_data)
                if y_pred is None or np.any(np.isnan(y_pred)):
                    continue

                mse = np.mean((y - y_pred) ** 2)
                # BIC-like score: fit - complexity penalty
                n = len(y)
                bic = -n * np.log(mse + 1e-12) - n_params * np.log(n)

                if bic > best_score:
                    best_score = bic
                    best = {
                        "target": target,
                        "parents": candidate_parents,
                        "expression": expr,
                        "latex": expr_latex,
                        "n_params": n_params,
                        "mse": mse,
                        "r2": 1 - mse / np.var(y) if np.var(y) > 0 else 0,
                        "bic": bic,
                    }
            except Exception:
                continue

        if best is None:
            return {"error": "No valid expression found"}

        # Check against known physics
        matching_law = self._match_known_law(best, target, candidate_parents)

        best["known_physics_match"] = matching_law

        return best

    def _generate_expressions(
        self, parents: List[str], max_degree: int, max_terms: int,
    ) -> List[Tuple[str, str, int]]:
        """Generate candidate symbolic expressions."""
        candidates = []

        for p in parents:
            # Linear: y = a * p
            candidates.append((f"{p}", f"{p}", 1))
            # Quadratic: y = a * p^2
            if max_degree >= 2:
                candidates.append((f"{p}**2", f"{p}^2", 1))
            # Inverse: y = a / p
            candidates.append((f"1 / ({p} + 1e-6)", f"1/{p}", 1))
            # Sqrt: y = a * sqrt(p)
            candidates.append((f"({p} + 1e-6)**0.5", f"\\sqrt{{{p}}}", 1))

        # Pairwise combinations
        if len(parents) >= 2 and max_terms >= 2:
            for i in range(len(parents)):
                for j in range(i + 1, len(parents)):
                    pi, pj = parents[i], parents[j]
                    # Product: y = a * pi * pj
                    candidates.append((f"{pi} * {pj}", f"{pi} \\cdot {pj}", 2))
                    # Ratio: y = a * pi / pj
                    candidates.append((f"{pi} / ({pj} + 1e-6)", f"{pi}/{pj}", 2))
                    # Sum: y = a*pi + b*pj
                    candidates.append((f"{pi} + {pj}", f"{pi} + {pj}", 2))

        # Polynomials with multiple terms
        if max_terms >= 3 and len(parents) >= 2:
            # Quadratic with interactions
            for i in range(len(parents)):
                for j in range(i, len(parents)):
                    pi, pj = parents[i], parents[j]
                    candidates.append((
                        f"{pi} + {pj} + {pi}*{pj}",
                        f"{pi} + {pj} + {pi}{pj}",
                        3
                    ))

        return candidates

    def _evaluate_expression(
        self, expr: str, data: Dict[str, np.ndarray],
    ) -> Optional[np.ndarray]:
        """Evaluate a symbolic expression on data."""
        # Replace variable names with underscore-prefixed versions for safe eval
        safe_expr = expr
        for var in data:
            safe_expr = safe_expr.replace(var, f"data['{var}']")
        try:
            return eval(safe_expr, {"data": data, "np": np})
        except Exception:
            return None

    def _match_known_law(
        self, discovered: Dict, target: str, parents: List[str],
    ) -> Optional[str]:
        """Check if discovered formula matches a known physical law."""
        for law in self.physics.list_all():
            if target in law.outputs and set(law.inputs) == set(parents):
                return law.name
        return None


# ═══════════════════════════════════════════════════════════════════
#  Convenience: Full Physics Pipeline
# ═══════════════════════════════════════════════════════════════════

def physics_causal_pipeline(
    data: np.ndarray,
    var_names: List[str],
    treatment: str,
    outcome: str,
    domain: str = "mechanics",
) -> Dict:
    """
    Run the full physics-informed causal pipeline:

    1. Causal discovery → get DAG skeleton
    2. PhysicsLibrary → find relevant laws, constrain DAG
    3. PhysicsInformedSCM → replace equations with physics formulas
    4. SymbolicPhysicsDiscovery → discover unknown formulas
    5. Counterfactual → with physics validation
    """
    from .discovery import pc_algorithm
    from .identification import identify_effect
    from .scm import linear_scm

    physics = PhysicsLibrary()

    # Step 1: Causal discovery
    try:
        dag_raw = pc_algorithm(data, var_names, alpha=0.05, max_cond_size=3)
    except ValueError:
        # Cycle detected — fall back to a minimal DAG
        from .graph import CausalDAG
        dag_raw = CausalDAG(var_names, [])

    # Step 2: Physics-constrained DAG
    pc_dag = PhysicsInformedCausalGraph(dag_raw, physics)
    dag = pc_dag.to_physics_corrected_dag()

    # Step 3: Build SCM with physics equations
    # Start with linear SCM, physics will override relevant equations
    scm_raw = linear_scm(dag, {}, noise_std=0.1)
    scm = PhysicsInformedSCM(scm_raw, physics)

    # Step 4: Discover unknown formulas (for non-physics variables)
    discoverer = SymbolicPhysicsDiscovery(physics)
    discoveries = {}
    for v in dag.variables:
        if v not in scm.physics_equations and dag.parents(v):
            disc = discoverer.discover_formula(
                data, var_names, v, list(dag.parents(v)),
                max_degree=2, max_terms=3,
            )
            if "error" not in disc:
                discoveries[v] = disc

    # Step 5: Effect identification
    ident = identify_effect(dag, treatment, outcome)

    # Step 6: Counterfactual (simple)
    cf = None
    if ident.adjustment_set is not None:
        sample_obs = {v: float(data[0, i]) for i, v in enumerate(var_names)}
        try:
            cf = scm.counterfactual_with_physics(
                sample_obs,
                {treatment: float(data[:, list(var_names).index(treatment)].mean())},
                outcome,
            )
        except Exception:
            pass

    return {
        "raw_dag": dag_raw,
        "physics_corrected_dag": dag,
        "constraints": pc_dag.constraint_report(),
        "physics_equations": scm.physics_equations,
        "discovered_formulas": discoveries,
        "identification": {
            "method": ident.method,
            "adjustment_set": ident.adjustment_set,
            "identifiable": ident.identifiable,
        },
        "counterfactual": cf,
        "summary": _generate_physics_report(
            dag_raw, dag, pc_dag, scm, discoveries, ident,
        ),
    }


def _generate_physics_report(
    dag_raw, dag, pc_dag, scm, discoveries, ident,
) -> str:
    lines = [
        "=" * 60,
        "  PHYSICS-INFORMED CAUSAL ANALYSIS",
        "=" * 60,
        "",
        "── Causal Skeleton ──",
        f"  Data-driven: {dag_raw}",
        f"  Physics-corrected: {dag}",
        "",
        pc_dag.constraint_report(),
        "",
        "── Physics-Governed Equations ──",
    ]
    if scm.physics_equations:
        for var, law_name in scm.physics_equations.items():
            law = scm.physics.get(law_name)
            lines.append(f"  {var} ← {law.latex}")
    else:
        lines.append("  (none applied)")

    lines.append("")
    lines.append("── Discovered Formulas ──")
    if discoveries:
        for var, disc in discoveries.items():
            match = f" [matches {disc['known_physics_match']}]" if disc["known_physics_match"] else ""
            lines.append(
                f"  {var} = f({', '.join(disc['parents'])}) "
                f"→ ${disc['latex']}$  "
                f"(R²={disc['r2']:.3f}){match}"
            )
    else:
        lines.append("  (none discovered)")

    lines.append("")
    lines.append("── Causal Identification ──")
    lines.append(f"  Method: {ident.method}")
    lines.append(f"  Adjust for: {ident.adjustment_set}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Test
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  PHYSICS-INFORMED CAUSAL ENGINE")
    print("=" * 60)

    physics = PhysicsLibrary()
    print(f"\n{physics.summary()}")

    # ── Test: Pendulum system ──
    print(f"\n{'─'*60}")
    print("  Demo: Simple Pendulum")
    print(f"{'─'*60}")

    rng = np.random.default_rng(42)
    n = 1000
    Length = rng.uniform(0.5, 2.0, n)
    Gravity = np.full(n, 9.81)
    # Physics: T = 2π√(L/g)
    Period = 2 * np.pi * np.sqrt(Length / Gravity) + rng.normal(0, 0.02, n)
    # Spurious correlation
    Amplitude = rng.uniform(5, 30, n)

    data = np.column_stack([Length, Gravity, Period, Amplitude])
    var_names = ["Length", "Gravity", "Period", "Amplitude"]

    result = physics_causal_pipeline(
        data, var_names,
        treatment="Length", outcome="Period",
        domain="mechanics",
    )

    print(f"\n{result['summary']}")

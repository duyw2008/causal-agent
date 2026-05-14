"""
Structural Causal Model (SCM).

A tuple M = ⟨U, V, F, P(u)⟩ where:
  - U: exogenous (unobserved) variables
  - V: endogenous (observed) variables
  - F: structural equations  v_i = f_i(pa(v_i), u_i)
  - P(u): distribution over exogenous variables

This module supports:
  - Linear SCMs (coefficient-based)
  - Non-parametric SCMs (Python-callable structural functions)
  - Sampling from the joint distribution
  - Intervention (do) operation
  - Counterfactual inference (abduction → action → prediction)
"""

from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from .graph import CausalDAG


# ── structural equations ────────────────────────────────────────

class StructuralEquation:
    """A single structural equation:  v = f(parents, noise)."""

    def __init__(
        self,
        variable: str,
        func: Callable[..., Any],
        parents: List[str],
        noise_dist: Optional[Callable[[], float]] = None,
    ):
        self.variable = variable
        self.func = func        # func(parent_values..., noise)
        self.parents = parents
        self.noise_dist = noise_dist or (lambda: np.random.normal(0, 1))

    def evaluate(
        self, parent_values: Dict[str, Any], noise: Optional[float] = None
    ) -> Any:
        """Compute v = f(pa(v), u)."""
        args = [parent_values[p] for p in self.parents]
        if noise is None:
            noise = self.noise_dist()
        return self.func(*args, noise)

    def __repr__(self) -> str:
        return f"{self.variable} = f({', '.join(self.parents)}, u_{self.variable})"


def linear_eq(coeffs: Dict[str, float], intercept: float = 0.0):
    """Factory for linear structural equations.

    v = intercept + Σ coeffs[p] * p + noise
    """
    keys = list(coeffs.keys())
    values = [coeffs[k] for k in keys]

    def _eq(*args, noise: float = 0.0) -> float:
        parent_vals = args[:len(keys)]
        return intercept + sum(c * v for c, v in zip(values, parent_vals)) + noise

    return _eq, keys


def nonlinear_eq(func: callable, parents: List[str]):
    """
    Factory for non-linear structural equations with additive noise.

    v = func(parents) + noise

    Parameters
    ----------
    func : callable
        f(*parent_values) → deterministic part
    parents : list of str
        Variable names of the parents (in order for func arguments)

    Example
    -------
    # a = F/m + noise
    eq, parents = nonlinear_eq(lambda f, m: f / max(m, 0.001), ["Force", "Mass"])

    # Logistic: P(Y=1) = sigmoid(βX + γZ) + noise
    eq, parents = nonlinear_eq(lambda x, z: 1/(1+np.exp(-(0.5*x+0.3*z))), ["X", "Z"])
    """
    def _eq(*args, noise: float = 0.0) -> float:
        parent_vals = args[:len(parents)]
        return func(*parent_vals) + noise

    return _eq, parents


def multiplicative_noise_eq(func: callable, parents: List[str]):
    """
    Factory for non-linear equations with multiplicative noise.

    v = func(parents) * exp(noise)

    Useful for log-normal models, growth rates, etc.
    Abduction: noise = log(v / func(parents))
    """
    import numpy as np

    def _eq(*args, noise: float = 0.0) -> float:
        parent_vals = args[:len(parents)]
        return func(*parent_vals) * np.exp(noise)

    return _eq, parents


# ── SCM ─────────────────────────────────────────────────────────

class SCM:
    """Structural Causal Model.

    Parameters
    ----------
    dag : CausalDAG
        The causal graph.
    equations : list of StructuralEquation
        One equation per endogenous variable.
    """

    def __init__(self, dag: CausalDAG, equations: List[StructuralEquation]):
        self.dag = dag
        # Index equations by variable name
        self._eqs: Dict[str, StructuralEquation] = {
            eq.variable: eq for eq in equations
        }
        # Verify every endogenous variable has an equation
        for v in dag.variables:
            if v not in self._eqs:
                raise ValueError(f"No equation for variable '{v}'")
        # Verify equation parents match DAG
        for v, eq in self._eqs.items():
            if set(eq.parents) != dag.parents(v):
                raise ValueError(
                    f"Parents mismatch for '{v}': "
                    f"DAG={dag.parents(v)}, eq={eq.parents}"
                )

    @property
    def variables(self) -> List[str]:
        return self.dag.variables

    @property
    def equations(self) -> Dict[str, StructuralEquation]:
        return dict(self._eqs)

    # ── sampling ─────────────────────────────────────────────

    def sample(
        self, n: int = 1, seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Draw n samples from the observational distribution.

        Returns a dict mapping variable names to (n,) arrays.
        """
        if seed is not None:
            np.random.seed(seed)

        order = self.dag.topological_order()
        data: Dict[str, np.ndarray] = {}
        noises: Dict[str, np.ndarray] = {}

        # Generate all noise first
        for v in order:
            eq = self._eqs[v]
            noises[v] = np.array([eq.noise_dist() for _ in range(n)])

        # Evaluate in topological order
        for v in order:
            eq = self._eqs[v]
            if not eq.parents:
                # Exogenous
                data[v] = np.array([
                    eq.func(noise=noises[v][i])
                    for i in range(n)
                ])
            else:
                parent_data = {p: data[p] for p in eq.parents}
                data[v] = np.array([
                    eq.func(
                        *[parent_data[p][i] for p in eq.parents],
                        noise=noises[v][i],
                    )
                    for i in range(n)
                ])

        return data

    # ── intervention ─────────────────────────────────────────

    def intervene(
        self, interventions: Dict[str, Any]
    ) -> IntervenedSCM:
        """Return a modified SCM where variables are set to fixed values.

        This implements Pearl's do(X=x) operator by:
          1. Removing the structural equations for X
          2. Setting X to the constant value x
        """
        return IntervenedSCM(self, interventions)

    # ── counterfactual ───────────────────────────────────────

    def counterfactual(
        self,
        observed: Dict[str, float],
        intervention: Dict[str, Any],
        target: str,
    ) -> float:
        """
        Three-step counterfactual inference:

        1. Abduction: infer noise values u from observed data
        2. Action: apply intervention do(X=x) to the model
        3. Prediction: compute the target under the modified model
        """
        # Step 1: Abduction — infer noise from observations
        order = self.dag.topological_order()
        noises: Dict[str, float] = {}
        computed: Dict[str, float] = {}

        for v in order:
            eq = self._eqs[v]
            if not eq.parents:
                # Exogenous: noise = observed - structural part
                noises[v] = observed[v] - eq.func(noise=0.0)  # f(noise=0) for exog
                noises[v] = observed[v]
                computed[v] = observed[v]
            else:
                parent_vals = [computed[p] for p in eq.parents]
                # Compute what the deterministic part gives
                # v = f(parents) + noise
                # noise = v - f(parents, 0)
                deterministic = eq.func(*parent_vals, noise=0.0)
                noises[v] = observed[v] - deterministic
                computed[v] = observed[v]

        # Step 2 & 3: Apply intervention and recompute
        mod = IntervenedSCM(self, intervention)
        return mod._compute_single(target, computed, noises)

    # ── display ──────────────────────────────────────────────

    def __repr__(self) -> str:
        eqs = [str(eq) for eq in self._eqs.values()]
        return f"SCM({len(eqs)} variables)\n" + "\n".join(f"  {e}" for e in eqs)

    def summary(self) -> str:
        lines = [f"SCM with {len(self._eqs)} endogenous variables:"]
        for v in self.dag.topological_order():
            eq = self._eqs[v]
            lines.append(f"  {v} = f({', '.join(eq.parents)}, u_{v})")
        return "\n".join(lines)


class IntervenedSCM:
    """An SCM under intervention do(X=x)."""

    def __init__(self, base: SCM, interventions: Dict[str, Any]):
        self.base = base
        self.interventions = interventions
        self._fixed = set(interventions.keys())
        # Modified equations: remove equations for intervened variables
        self._eqs = {
            v: eq for v, eq in base._eqs.items()
            if v not in self._fixed
        }

    def _compute_single(
        self,
        target: str,
        pre_values: Dict[str, float],
        noises: Dict[str, float],
    ) -> float:
        """Compute a single value under intervention.

        Recomputes ALL variables in topological order. Variables
        set by intervention get their fixed values. All others are
        recomputed using their structural equations with the
        (possibly changed) parent values and the original noise
        (from abduction). This correctly propagates intervention
        effects through the entire downstream graph.
        """
        order = self.base.dag.topological_order()
        values: Dict[str, float] = {}

        for v in order:
            if v in self._fixed:
                values[v] = self.interventions[v]
            else:
                eq = self.base._eqs[v]
                if not eq.parents:
                    # Exogenous: use original noise
                    values[v] = noises.get(v, 0.0)
                else:
                    # Recompute using current parent values + original noise
                    parent_vals = [values[p] for p in eq.parents]
                    values[v] = eq.func(*parent_vals, noise=noises.get(v, 0.0))

        return values.get(target, 0.0)

    def sample(self, n: int = 1, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Sample from the intervened distribution."""
        if seed is not None:
            np.random.seed(seed)

        order = self.base.dag.topological_order()
        data: Dict[str, np.ndarray] = {}

        for v in order:
            if v in self._fixed:
                data[v] = np.full(n, self.interventions[v])
            else:
                eq = self.base._eqs[v]
                noises = np.array([eq.noise_dist() for _ in range(n)])
                if not eq.parents:
                    data[v] = np.array([eq.func(noise=noises[i]) for i in range(n)])
                else:
                    data[v] = np.array([
                        eq.func(
                            *[data[p][i] for p in eq.parents],
                            noise=noises[i],
                        )
                        for i in range(n)
                    ])

        return data


# ── factory helpers ─────────────────────────────────────────────

def linear_scm(
    dag: CausalDAG,
    coefficients: Dict[str, Dict[str, float]],
    intercepts: Optional[Dict[str, float]] = None,
    noise_std: float = 1.0,
) -> SCM:
    """Create a linear SCM from coefficients.

    Parameters
    ----------
    dag : CausalDAG
    coefficients : dict
        coefficients[v][p] = coefficient from parent p to v.
        e.g., {"Y": {"X": 2.0, "Z": -1.0}}
    intercepts : optional dict
        Intercepts for each variable.
    noise_std : float
        Standard deviation of Gaussian noise for all variables.
    """
    intercepts = intercepts or {}
    equations = []

    for v in dag.variables:
        parents = list(dag.parents(v))
        coeffs = coefficients.get(v, {})
        icpt = intercepts.get(v, 0.0)

        parent_names = sorted(parents)
        coeff_values = [coeffs.get(p, 0.0) for p in parent_names]

        def make_eq(pnames, cvals, icpt):
            def _eq(*args, noise: float = 0.0) -> float:
                return icpt + sum(c * a for c, a in zip(cvals, args)) + noise
            return _eq

        eq_func = make_eq(parent_names, coeff_values, icpt)
        equations.append(
            StructuralEquation(
                v, eq_func, parent_names,
                noise_dist=lambda: np.random.normal(0, noise_std),
            )
        )

    return SCM(dag, equations)


def nonlinear_scm(
    dag: CausalDAG,
    funcs: Dict[str, Tuple[callable, List[str]]],
    noise_std: float = 0.1,
    noise_type: str = "additive",
) -> SCM:
    """
    Create a non-linear SCM from arbitrary Python functions.

    Parameters
    ----------
    dag : CausalDAG
    funcs : dict
        {variable: (function, parent_names)}
        function must be f(*parent_values) → deterministic_part
        e.g., {"Y": (lambda x: x**2 + 3, ["X"])}
    noise_std : float
        Standard deviation of additive Gaussian noise
    noise_type : str
        "additive" — v = f(parents) + noise
        "multiplicative" — v = f(parents) * exp(noise)

    Returns
    -------
    SCM

    Example
    -------
    >>> dag = CausalDAG(["X", "Y"], [("X", "Y")])
    >>> scm = nonlinear_scm(dag, {
    ...     "Y": (lambda x: 0.5 * x**2 + np.sin(x), ["X"])
    ... })
    >>> data = scm.sample(1000)
    >>> cf = scm.counterfactual({"X": 2.0, "Y": 5.0}, {"X": 0.0}, "Y")
    """
    equations = []

    for v in dag.variables:
        parents = list(dag.parents(v))
        if v in funcs:
            fn, expected_parents = funcs[v]
            # Verify parents match DAG
            if set(expected_parents) != set(parents):
                raise ValueError(
                    f"Parent mismatch for {v}: DAG says {parents}, "
                    f"function expects {expected_parents}"
                )

            if noise_type == "multiplicative":
                eq_func, _ = multiplicative_noise_eq(fn, expected_parents)
            else:
                eq_func, _ = nonlinear_eq(fn, expected_parents)

            equations.append(
                StructuralEquation(
                    v, eq_func, expected_parents,
                    noise_dist=lambda: np.random.normal(0, noise_std),
                )
            )
        else:
            # Exogenous variable (no parents)
            if noise_type == "multiplicative":
                eq_func, _ = multiplicative_noise_eq(lambda: 0.0, [])
            else:
                eq_func, _ = nonlinear_eq(lambda: 0.0, [])
            equations.append(
                StructuralEquation(
                    v, eq_func, [],
                    noise_dist=lambda: np.random.normal(0, noise_std),
                )
            )

    return SCM(dag, equations)


# ── tests ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simple chain: X → M → Y
    # X ~ N(0,1)
    # M = 0.5*X + u_M
    # Y = 0.8*M + u_Y
    dag = CausalDAG(["X", "M", "Y"], [("X", "M"), ("M", "Y")])

    scm = linear_scm(
        dag,
        coefficients={"M": {"X": 0.5}, "Y": {"M": 0.8}},
        noise_std=0.1,
    )

    # Observational sampling
    data = scm.sample(10000, seed=42)
    print(f"E[Y] = {np.mean(data['Y']):.3f}")

    # Intervention: do(M=1)
    intv = scm.intervene({"M": 1.0})
    intv_data = intv.sample(10000, seed=42)
    print(f"E[Y | do(M=1)] = {np.mean(intv_data['Y']):.3f}")
    print(f"Expected: 0.8 * 1.0 = 0.8")

    # Counterfactual
    cf = scm.counterfactual(
        observed={"X": 2.0, "M": 1.2, "Y": 1.0},
        intervention={"M": 0.0},
        target="Y",
    )
    print(f"Counterfactual Y (M=0, given X=2): {cf:.3f}")

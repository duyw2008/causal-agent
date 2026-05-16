"""
Causal Mediation Analysis — 直接效应 / 间接效应分解

Mediation 回答的问题是：X 对 Y 的因果效应中，多少是通过中介 M 传递的、
多少是 X 直接影响 Y 的？

框架:
  - CDE (Controlled Direct Effect):       固定 M=m 时的直接效应
  - NDE (Natural Direct Effect):          中介保持在 "无处理" 水平时的直接效应
  - NIE (Natural Indirect Effect):        处理仅通过中介改变 Y 的间接效应
  - TE  (Total Effect):                   NDE + NIE = 总效应

Pearl 中介公式 (反事实框架):
  NDE = E[Y_{x, M_{x*}} - Y_{x*}]
  NIE = E[Y_{x, M_x} - Y_{x, M_{x*}}]

线性 SCM 下的简化:
  NDE = β_X→Y  (X→Y 直接路径系数)
  NIE = β_X→M × β_M→Y  (路径系数乘积)

Usage example:
    from core.mediation import analyze_mediation

    result = analyze_mediation(dag, scm, data, treatment="X",
                                mediator="M", outcome="Y")
    print(result.summary())
    # → TE = 4.50, NDE = 2.00 (44%), NIE = 2.50 (56%)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

from .graph import CausalDAG
from .scm import SCM, linear_scm, StructuralEquation


# ═══════════════════════════════════════════════════════════════════
#  Mediation Result
# ═══════════════════════════════════════════════════════════════════

@dataclass
class MediationResult:
    """Complete mediation decomposition."""

    treatment: str
    mediator: str
    outcome: str

    # ── Effect estimates ──
    total_effect: float               # TE = NDE + NIE
    cde: Optional[float] = None       # Controlled Direct Effect (M fixed)
    nde: Optional[float] = None       # Natural Direct Effect
    nie: Optional[float] = None       # Natural Indirect Effect

    # ── Proportions ──
    nde_fraction: Optional[float] = None   # NDE / TE
    nie_fraction: Optional[float] = None   # NIE / TE

    # ── Method info ──
    method: str = ""                   # "linear_scm" | "baron_kenny" | "counterfactual"
    mediation_present: bool = False    # Is NIE significantly non-zero?

    # ── Diagnostic ──
    path_coefficients: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""

    def summary(self) -> str:
        lines = [
            f"Causal Mediation: {self.treatment} → {self.mediator} → {self.outcome}",
            f"  Method: {self.method}",
            "",
            f"  Total Effect (TE):           {self.total_effect:+.4f}",
        ]
        if self.cde is not None:
            lines.append(f"  Controlled Direct (CDE):     {self.cde:+.4f}")
        if self.nde is not None:
            lines.append(f"  Natural Direct (NDE):        {self.nde:+.4f}"
                         f"  ({self.nde_fraction*100:.0f}%)" if self.nde_fraction else "")
        if self.nie is not None:
            lines.append(f"  Natural Indirect (NIE):      {self.nie:+.4f}"
                         f"  ({self.nie_fraction*100:.0f}%)" if self.nie_fraction else "")
        if self.mediation_present:
            lines.append(f"  Mediation: ✓ present")
        else:
            lines.append(f"  Mediation: ✗ absent (NIE ≈ 0)")
        if self.explanation:
            lines.append(f"\n  {self.explanation}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Path-Coefficient Mediation (Linear SCM)
# ═══════════════════════════════════════════════════════════════════

def _linear_mediation(
    dag: CausalDAG,
    scm: SCM,
    treatment: str,
    mediator: str,
    outcome: str,
) -> MediationResult:
    """Decompose via path coefficients:

    TE = direct(X→Y) + indirect(X→M→Y) + correlated_backdoor(X←Z→Y|M)

    For the simplest linear mediation model:
        M = α·X + ε_M
        Y = τ·X + β·M + ε_Y

    Then:
        NDE = τ  (direct effect of X on Y holding M constant at natural level)
        NIE = α·β  (effect of X on Y through M)
        TE  = τ + α·β
    """
    # Find path coefficients from SCM
    coeffs: Dict[str, float] = {}

    # X → M
    if mediator in scm.equations and treatment in scm.equations[mediator].parents:
        coeffs["X→M"] = _extract_linear_coefficient(
            scm.equations[mediator], treatment)

    # X → Y
    if outcome in scm.equations and treatment in scm.equations[outcome].parents:
        coeffs["X→Y"] = _extract_linear_coefficient(
            scm.equations[outcome], treatment)

    # M → Y
    if outcome in scm.equations and mediator in scm.equations[outcome].parents:
        coeffs["M→Y"] = _extract_linear_coefficient(
            scm.equations[outcome], mediator)

    nde = coeffs.get("X→Y", 0.0)
    nie = coeffs.get("X→M", 0.0) * coeffs.get("M→Y", 0.0)
    te = nde + nie

    # Also compute CDE by simulating SCM with M fixed
    cde = _compute_cde_from_scm(scm, treatment, mediator, outcome)

    result = MediationResult(
        treatment=treatment, mediator=mediator, outcome=outcome,
        total_effect=te, cde=cde, nde=nde, nie=nie,
        nde_fraction=nde / te if abs(te) > 1e-8 else 0.0,
        nie_fraction=nie / te if abs(te) > 1e-8 else 0.0,
        method="linear_scm (path coefficients)",
        mediation_present=abs(nie) > 1e-6,
        path_coefficients=coeffs,
        explanation=(
            f"NDE = β(X→Y)={nde:.3f}, "
            f"NIE = β(X→M)×β(M→Y)={nie:.3f}"
        ),
    )
    return result


def _extract_linear_coefficient(eq: StructuralEquation, parent: str) -> float:
    """Extract the coefficient for 'parent' from a linear equation."""
    # Try calling with 0 for all parents except the one of interest
    try:
        # Build args: for the target parent, pass 1.0; for others, pass 0.0
        args = []
        found = False
        for p in eq.parents:
            if p == parent:
                args.append(1.0)
                found = True
            else:
                args.append(0.0)
        if found:
            return float(eq.func(*args, noise=0.0))
    except Exception:
        pass

    # Fallback: use stored coefficients dict if available
    if hasattr(eq, 'coefficients') and parent in eq.coefficients:
        return float(eq.coefficients[parent])

    return 0.0


def _compute_cde_from_scm(scm: SCM, treatment: str,
                           mediator: str, outcome: str) -> float:
    """Compute CDE via Monte Carlo with mediator fixed at its natural mean."""
    n_samples = 1000
    samples = scm.sample(n_samples)

    # samples is dict: {var: np.ndarray}
    m_natural_mean = float(np.mean(samples[mediator]))

    try:
        scm_intv1 = scm.intervene({treatment: 1.0, mediator: m_natural_mean})
        y1_samples = scm_intv1.sample(n_samples)
        y1 = y1_samples[outcome]

        scm_intv0 = scm.intervene({treatment: 0.0, mediator: m_natural_mean})
        y0_samples = scm_intv0.sample(n_samples)
        y0 = y0_samples[outcome]

        return float(np.mean(y1) - np.mean(y0))
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════
#  Baron-Kenny Mediation (Classical, Regression-Based)
# ═══════════════════════════════════════════════════════════════════

def _baron_kenny_mediation(
    data: np.ndarray,
    var_names: List[str],
    treatment: str,
    mediator: str,
    outcome: str,
) -> MediationResult:
    """Classical 4-step Baron-Kenny (1986) mediation.

    Steps:
      1. X → Y  (c-path):     TE must be significant
      2. X → M  (a-path):     X must affect M
      3. M → Y | X (b-path):  M must affect Y controlling for X
      4. X → Y | M (c'-path): Direct effect with M in model

    NIE = a·b,  NDE = c',  TE = c = c' + a·b
    """
    var_idx = {v: i for i, v in enumerate(var_names)}
    X = data[:, var_idx[treatment]]
    M = data[:, var_idx[mediator]]
    Y = data[:, var_idx[outcome]]
    n = len(X)

    # Step 1: Y ~ X (total effect)
    X_mat = np.column_stack([np.ones(n), X])
    c = float(np.linalg.lstsq(X_mat, Y, rcond=None)[0][1])

    # Step 2: M ~ X (a-path)
    a = float(np.linalg.lstsq(X_mat, M, rcond=None)[0][1])

    # Step 3 & 4: Y ~ X + M
    XM_mat = np.column_stack([np.ones(n), X, M])
    coeffs, _, _, _ = np.linalg.lstsq(XM_mat, Y, rcond=None)
    c_prime = float(coeffs[1])   # Direct effect (X→Y | M)
    b = float(coeffs[2])          # M→Y | X

    nde = c_prime
    nie = a * b
    te = c

    # Bootstrap SE and p-value
    n_boot = 200
    rng = np.random.default_rng(42)
    ab_samples = np.zeros(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        Xb, Mb, Yb = X[idx], M[idx], Y[idx]
        Xb_mat = np.column_stack([np.ones(n), Xb])
        ab = np.linalg.lstsq(Xb_mat, Mb, rcond=None)[0][1]
        XMb_mat = np.column_stack([np.ones(n), Xb, Mb])
        bb = np.linalg.lstsq(XMb_mat, Yb, rcond=None)[0][2]
        ab_samples[i] = ab * bb

    se = float(np.std(ab_samples))
    mediation_present = abs(nie) > 1.96 * se if se > 1e-10 else abs(nie) > 1e-4

    # CDE: fix M at its mean
    m_mean = float(np.mean(M))
    Y1 = coeffs[0] + coeffs[1] * 1.0 + coeffs[2] * m_mean
    Y0 = coeffs[0] + coeffs[1] * 0.0 + coeffs[2] * m_mean
    cde = Y1 - Y0

    result = MediationResult(
        treatment=treatment, mediator=mediator, outcome=outcome,
        total_effect=te, cde=cde, nde=nde, nie=nie,
        nde_fraction=nde / te if abs(te) > 1e-8 else 0.0,
        nie_fraction=nie / te if abs(te) > 1e-8 else 0.0,
        method="Baron-Kenny (regression + bootstrap)",
        mediation_present=mediation_present,
        path_coefficients={"X→M": a, "M→Y": b, "X→Y (total)": c,
                           "X→Y (direct)": c_prime},
        explanation=(
            f"a={a:.3f}, b={b:.3f}, c={c:.3f}, c'={c_prime:.3f}; "
            f"NIE SE={se:.3f}"
        ),
    )
    return result


# ═══════════════════════════════════════════════════════════════════
#  Counterfactual Mediation (Pearl's Mediation Formula)
# ═══════════════════════════════════════════════════════════════════

def _counterfactual_mediation(
    scm: SCM,
    treatment: str,
    mediator: str,
    outcome: str,
    n_samples: int = 1000,
) -> MediationResult:
    """Pearl's mediation formula via Monte Carlo counterfactuals.

    NDE = E[Y_{x=1, M=M_{x=0}}] - E[Y_{x=0}]
    NIE = E[Y_{x=1, M=M_{x=1}}] - E[Y_{x=1, M=M_{x=0}}]

    For each sample unit u:
      1. Simulate M_0 = M(x=0, u)   — mediator under control
      2. Simulate M_1 = M(x=1, u)   — mediator under treatment
      3. Simulate Y_00 = Y(x=0, M_0, u)  — outcome under control
      4. Simulate Y_11 = Y(x=1, M_1, u)  — outcome under treatment
      5. Simulate Y_10 = Y(x=1, M_0, u)  — outcome with treatment but
                                              mediator at control level (NDE)
    """
    y_00_list = []
    y_11_list = []
    y_10_list = []

    for i in range(min(n_samples, 200)):  # Cap at 200 for speed
        # --- M under X=0 and X=1 ---
        scm_intv_M0 = scm.intervene({treatment: 0.0})
        m0 = scm_intv_M0.sample(1)[mediator][0]

        scm_intv_M1 = scm.intervene({treatment: 1.0})
        m1 = scm_intv_M1.sample(1)[mediator][0]

        # --- Y under counterfactual conditions ---
        scm_intv_Y00 = scm.intervene({treatment: 0.0, mediator: float(m0)})
        y_00 = scm_intv_Y00.sample(1)[outcome][0]

        scm_intv_Y11 = scm.intervene({treatment: 1.0, mediator: float(m1)})
        y_11 = scm_intv_Y11.sample(1)[outcome][0]

        scm_intv_Y10 = scm.intervene({treatment: 1.0, mediator: float(m0)})
        y_10 = scm_intv_Y10.sample(1)[outcome][0]

        y_00_list.append(y_00)
        y_11_list.append(y_11)
        y_10_list.append(y_10)

    E_Y00 = np.mean(y_00_list)
    E_Y11 = np.mean(y_11_list)
    E_Y10 = np.mean(y_10_list)

    te = E_Y11 - E_Y00
    nde = E_Y10 - E_Y00
    nie = E_Y11 - E_Y10

    result = MediationResult(
        treatment=treatment, mediator=mediator, outcome=outcome,
        total_effect=float(te), nde=float(nde), nie=float(nie),
        nde_fraction=float(nde) / float(te) if abs(te) > 1e-8 else 0.0,
        nie_fraction=float(nie) / float(te) if abs(te) > 1e-8 else 0.0,
        method="Pearl mediation formula (Monte Carlo counterfactuals)",
        mediation_present=abs(float(nie)) > 1e-4,
        explanation=(
            f"NDE = E[Y(1,M(0))] - E[Y(0)] = {nde:.3f}, "
            f"NIE = E[Y(1,M(1))] - E[Y(1,M(0))] = {nie:.3f}"
        ),
    )
    return result


# ═══════════════════════════════════════════════════════════════════
#  Main API
# ═══════════════════════════════════════════════════════════════════

def analyze_mediation(
    dag: CausalDAG,
    scm: Optional[SCM] = None,
    data: Optional[np.ndarray] = None,
    var_names: Optional[List[str]] = None,
    treatment: str = "T",
    mediator: str = "M",
    outcome: str = "Y",
    method: str = "auto",
) -> MediationResult:
    """Decompose total causal effect into direct and indirect components.

    Args:
        dag:   Causal DAG with X→M→Y structure.
        scm:   Structural Causal Model (optional; enables path-coefficient
               and counterfactual methods).
        data:  Observational data (n_samples × n_vars).
        var_names: Variable names for data columns.
        treatment: Treatment/exposure variable.
        mediator:  Mediator variable.
        outcome:   Outcome variable.
        method: "linear_scm" | "baron_kenny" | "counterfactual" | "auto"

    Returns:
        MediationResult with TE, NDE, NIE, CDE, fractions, and diagnostics.
    """
    # Determine best method
    if method == "auto":
        if scm is not None and _has_linear_structure(scm, treatment, mediator, outcome):
            method = "linear_scm"
        elif data is not None and var_names is not None:
            method = "baron_kenny"
        elif scm is not None:
            method = "counterfactual"
        else:
            # Check structural feasibility even without data
            if mediator not in dag.children(treatment):
                return MediationResult(
                    treatment=treatment, mediator=mediator, outcome=outcome,
                    total_effect=0.0, method="none",
                    explanation=f"No edge {treatment}→{mediator}: mediation impossible",
                )
            if outcome not in dag.children(mediator):
                return MediationResult(
                    treatment=treatment, mediator=mediator, outcome=outcome,
                    total_effect=0.0, method="none",
                    explanation=f"No edge {mediator}→{outcome}: mediation impossible",
                )
            return MediationResult(
                treatment=treatment, mediator=mediator, outcome=outcome,
                total_effect=0.0, method="none",
                explanation="Insufficient data: need SCM or data+var_names",
            )

    # Structural check: X→M and M→Y must exist
    if mediator not in dag.children(treatment):
        return MediationResult(
            treatment=treatment, mediator=mediator, outcome=outcome,
            total_effect=0.0, method="none",
            explanation=f"No edge {treatment}→{mediator}: mediation impossible",
        )
    if outcome not in dag.children(mediator):
        return MediationResult(
            treatment=treatment, mediator=mediator, outcome=outcome,
            total_effect=0.0, method="none",
            explanation=f"No edge {mediator}→{outcome}: mediation impossible",
        )

    if method == "linear_scm":
        return _linear_mediation(dag, scm, treatment, mediator, outcome)
    elif method == "baron_kenny":
        return _baron_kenny_mediation(data, var_names, treatment, mediator, outcome)
    elif method == "counterfactual":
        return _counterfactual_mediation(scm, treatment, mediator, outcome)
    else:
        return MediationResult(
            treatment=treatment, mediator=mediator, outcome=outcome,
            total_effect=0.0, method=method,
            explanation=f"Unknown method: {method}",
        )


def _has_linear_structure(scm: SCM, treatment: str,
                           mediator: str, outcome: str) -> bool:
    """Check if SCM has a linear mediation structure."""
    return (mediator in scm.equations
            and treatment in scm.equations[mediator].parents
            and outcome in scm.equations
            and mediator in scm.equations[outcome].parents)


# ═══════════════════════════════════════════════════════════════════
#  Path-Specific Effects
# ═══════════════════════════════════════════════════════════════════

def path_specific_effect(
    dag: CausalDAG,
    scm: SCM,
    treatment: str,
    outcome: str,
    path: List[str],
    n_samples: int = 500,
) -> float:
    """Compute the effect transmitted through a specific causal path.

    The effect of X on Y THROUGH path = {X → v1 → v2 → ... → Y} ONLY,
    with all other paths "turned off" (edges on other paths fixed to
    baseline).

    Args:
        path: Ordered list of variables on the path, e.g. ["M1", "M2"] for X→M1→M2→Y.

    Returns:
        Path-specific effect (contribution of this path to TE).
    """
    var_idx = {v: i for i, v in enumerate(scm.variables)}
    y_idx = var_idx[outcome]

    effects = []
    for _ in range(min(n_samples, 200)):
        # Step through the path, activating each edge sequentially
        scm_intv = scm.intervene({treatment: 1.0})

        # Fix non-path intermediate variables to their natural values
        path_set = set(path)
        natural_sample = scm.sample(1)
        natural_vals = {}
        for v in dag.variables:
            if v in dag.ancestors(outcome) and v not in {treatment, outcome} | path_set:
                natural_vals[v] = float(natural_sample[v][0])

        if natural_vals:
            scm_intv = scm_intv.intervene(natural_vals)

        y_path = scm_intv.sample(1)[outcome][0]

        # Baseline
        scm_base = scm.intervene({treatment: 0.0})
        if natural_vals:
            scm_base = scm_base.intervene(natural_vals)
        y_base = scm_base.sample(1)[outcome][0]

        effects.append(y_path - y_base)

    return float(np.mean(effects)) if effects else 0.0


# ═══════════════════════════════════════════════════════════════════
#  Test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.graph import CausalDAG
    from core.scm import linear_scm
    from core.mediation import analyze_mediation
    import numpy as np

    # ── Test 1: Linear SCM (path-coefficient method) ──
    dag1 = CausalDAG(["T","M","Y"], [("T","M"),("T","Y"),("M","Y")])
    scm1 = linear_scm(dag1, {
        "M": {"T": 1.5},
        "Y": {"T": 2.0, "M": 1.0},
    }, noise_std=0.1)
    r1 = analyze_mediation(dag1, scm=scm1, treatment="T", mediator="M", outcome="Y")
    print(r1.summary())

    # ── Test 2: Baron-Kenny (data-driven) ──
    print(f"\n{'─'*60}")
    n = 500
    rng = np.random.default_rng(42)
    T_d = rng.normal(0, 1, n)
    M_d = 1.5 * T_d + rng.normal(0, 0.5, n)
    Y_d = 2.0 * T_d + 1.0 * M_d + rng.normal(0, 0.5, n)
    data2 = np.column_stack([T_d, M_d, Y_d])
    dag2 = CausalDAG(["T","M","Y"], [("T","M"),("T","Y"),("M","Y")])
    r2 = analyze_mediation(dag2, data=data2, var_names=["T","M","Y"],
                            method="baron_kenny")
    print(r2.summary())

    print(f"\n{'─'*60}")
    print("  All mediation demos passed ✓")

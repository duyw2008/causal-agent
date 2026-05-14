"""
Causal Effect Estimation — numerical ATE computation.

Given:   DAG → identified adjustment set Z
         Data → (n_samples, n_variables)
         Treatment T, Outcome Y

Output:  ATE estimate ± confidence interval

Estimators:
  - LinearRegression     : Y ~ T + Z           (fast, interpretable)
  - PropensityScoreMatching : match on P(T|Z)  (semi-parametric)
  - InverseProbabilityWeighting : weight by 1/P(T|Z) (semi-parametric)
  - DoublyRobust         : IPW + outcome model  (double protection)
  - Stratification       : strata by P(T|Z)     (simple, intuitive)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class CausalEstimate:
    """Result of a causal effect estimation."""

    ate: float                    # Average Treatment Effect
    std_error: float              # Standard error
    ci_lower: float               # 95% CI lower bound
    ci_upper: float               # 95% CI upper bound
    method: str                   # Estimator name
    n_samples: int
    adjustment_set: List[str]

    # Optional diagnostics
    propensity_overlap: Optional[float] = None  # fraction in common support
    smd_before: Optional[Dict[str, float]] = None  # standardized mean diff before
    smd_after: Optional[Dict[str, float]] = None   # standardized mean diff after
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def summary(self) -> str:
        lines = [
            f"Causal Effect Estimate ({self.method})",
            f"  ATE = {self.ate:.4f}",
            f"  95% CI = [{self.ci_lower:.4f}, {self.ci_upper:.4f}]",
            f"  SE = {self.std_error:.4f}",
            f"  N = {self.n_samples}",
        ]
        if self.adjustment_set:
            lines.append(f"  Adjusted for: {', '.join(self.adjustment_set)}")
        if self.warnings:
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        return "\n".join(lines)

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Is the effect statistically significant? (Wald test)"""
        z = abs(self.ate) / max(self.std_error, 1e-12)
        # Approximate: |z| > 1.96 → p < 0.05
        if alpha == 0.05:
            return z > 1.96
        if alpha == 0.01:
            return z > 2.576
        if alpha == 0.10:
            return z > 1.645
        # General case: normal tail approximation
        p_value = np.exp(-z**2 / 2) / (z * np.sqrt(2 * np.pi) + 1e-12)
        return p_value < alpha


# ═══════════════════════════════════════════════════════════════════
#  Helper: prepare data
# ═══════════════════════════════════════════════════════════════════

def _get_columns(
    data: np.ndarray,
    var_names: List[str],
    treatment: str,
    outcome: str,
    adjustment_set: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extract treatment, outcome, and confounder columns from data matrix."""
    name_to_idx = {v: i for i, v in enumerate(var_names)}

    t_col = data[:, name_to_idx[treatment]]
    y_col = data[:, name_to_idx[outcome]]

    if adjustment_set:
        z_idx = [name_to_idx[z] for z in adjustment_set]
        z_cols = data[:, z_idx]
    else:
        z_cols = None

    return t_col, y_col, z_cols


# ═══════════════════════════════════════════════════════════════════
#  1. Linear Regression Estimator
# ═══════════════════════════════════════════════════════════════════

def estimate_ate_linear(
    data: np.ndarray,
    var_names: List[str],
    treatment: str,
    outcome: str,
    adjustment_set: Optional[List[str]] = None,
) -> CausalEstimate:
    """
    Estimate ATE via linear regression: Y ~ β·T + γ·Z + ε.

    ATE = β  (the coefficient on T).

    Assumes: linearity, no interaction between T and Z.
    """
    t, y, z = _get_columns(data, var_names, treatment, outcome, adjustment_set)
    n = len(t)

    # Build design matrix: [intercept, T, Z1, Z2, ...]
    if z is not None:
        X = np.column_stack([np.ones(n), t, z])
    else:
        X = np.column_stack([np.ones(n), t])

    # OLS
    beta, residuals, rank, singular = np.linalg.lstsq(X, y, rcond=None)

    # Standard errors
    y_pred = X @ beta
    residuals = y - y_pred
    mse = np.sum(residuals ** 2) / (n - X.shape[1])
    # (X'X)^-1
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * mse)

    ate = beta[1]  # coefficient on T
    se_ate = se[1]

    ci_lower = ate - 1.96 * se_ate
    ci_upper = ate + 1.96 * se_ate

    warnings = []
    if rank < X.shape[1]:
        warnings.append("Design matrix is rank-deficient; estimates may be unstable")

    return CausalEstimate(
        ate=float(ate),
        std_error=float(se_ate),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        method="Linear Regression",
        n_samples=n,
        adjustment_set=adjustment_set or [],
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════
#  2. Propensity Score Matching (PSM)
# ═══════════════════════════════════════════════════════════════════

def estimate_ate_psm(
    data: np.ndarray,
    var_names: List[str],
    treatment: str,
    outcome: str,
    adjustment_set: Optional[List[str]] = None,
    n_neighbors: int = 1,
    caliper: Optional[float] = None,
) -> CausalEstimate:
    """
    Estimate ATE via propensity score matching.

    1. Estimate propensity score: P(T=1 | Z) via logistic regression
    2. Match each treated unit to nearest control unit(s) by propensity
    3. ATE = mean(Y_treated) - mean(Y_matched_control)

    For continuous T: binarize at median.
    """
    t, y, z = _get_columns(data, var_names, treatment, outcome, adjustment_set)
    n = len(t)

    # Binarize treatment if continuous
    if len(np.unique(t)) > 2:
        t_median = np.median(t)
        t_bin = (t > t_median).astype(float)
        warnings = [f"Treatment binarized at median ({t_median:.2f})"]
    else:
        t_bin = t.copy()
        warnings = []

    # Estimate propensity score
    if z is not None:
        ps = _logistic_propensity_score(z, t_bin)
    else:
        ps = np.full(n, t_bin.mean())

    # Matching
    treated_idx = np.where(t_bin == 1)[0]
    control_idx = np.where(t_bin == 0)[0]
    ps_treated = ps[treated_idx]
    ps_control = ps[control_idx]

    y_treated = y[treated_idx]
    y_control = y[control_idx]

    matched_control_y = []
    overlap_count = 0

    for i, t_idx in enumerate(treated_idx):
        # Distance to all controls
        dists = np.abs(ps_control - ps_treated[i])

        # Caliper
        if caliper is not None:
            valid = dists <= caliper
            if not valid.any():
                continue
            dists = dists[valid]
            c_idx_subset = control_idx[valid]
        else:
            c_idx_subset = control_idx

        # Find nearest neighbors
        if n_neighbors == 1:
            best = np.argmin(dists)
            matched_control_y.append(y_control[best])
        else:
            best_k = np.argsort(dists)[:n_neighbors]
            matched_control_y.append(y_control[best_k].mean())

        overlap_count += 1

    if len(matched_control_y) == 0:
        return CausalEstimate(
            ate=0.0, std_error=np.inf, ci_lower=-np.inf, ci_upper=np.inf,
            method="PSM (failed)", n_samples=n,
            adjustment_set=adjustment_set or [],
            warnings=["No matches found; check propensity score overlap"],
        )

    matched_y = np.array(matched_control_y)
    y_treated_matched = y_treated[:len(matched_y)]

    # ATE = mean difference
    diffs = y_treated_matched - matched_y
    ate = diffs.mean()
    se_ate = diffs.std() / np.sqrt(len(diffs))

    ci_lower = ate - 1.96 * se_ate
    ci_upper = ate + 1.96 * se_ate

    # Diagnostics
    ps_overlap = overlap_count / len(treated_idx) if len(treated_idx) > 0 else 0.0

    if ps_overlap < 0.8:
        warnings.append(
            f"Low propensity overlap ({ps_overlap:.1%}): "
            "treated and control distributions are very different"
        )

    # SMD before/after (simplified)
    smd_before = {}
    smd_after = {}
    if z is not None and adjustment_set:
        for j, z_name in enumerate(adjustment_set):
            z_col = z[:, j]
            smd_before[z_name] = _smd(z_col[t_bin == 1], z_col[t_bin == 0])

    return CausalEstimate(
        ate=float(ate),
        std_error=float(se_ate),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        method=f"PSM (k={n_neighbors})",
        n_samples=n,
        adjustment_set=adjustment_set or [],
        propensity_overlap=float(ps_overlap),
        smd_before=smd_before,
        smd_after=smd_after,
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════
#  3. Inverse Probability Weighting (IPW)
# ═══════════════════════════════════════════════════════════════════

def estimate_ate_ipw(
    data: np.ndarray,
    var_names: List[str],
    treatment: str,
    outcome: str,
    adjustment_set: Optional[List[str]] = None,
    stabilize: bool = True,
) -> CausalEstimate:
    """
    Estimate ATE via Inverse Probability Weighting.

    w_i = T_i / p_i + (1 - T_i) / (1 - p_i)    (unstabilized)
    w_i = [T_i / p_i] * P(T=1) + [(1-T_i)/(1-p_i)] * P(T=0)  (stabilized)

    ATE = (Σ w_i · T_i · Y_i) / (Σ w_i · T_i) - (Σ w_i · (1-T_i) · Y_i) / (Σ w_i · (1-T_i))
    """
    t, y, z = _get_columns(data, var_names, treatment, outcome, adjustment_set)
    n = len(t)

    # Binarize
    if len(np.unique(t)) > 2:
        t_median = np.median(t)
        t_bin = (t > t_median).astype(float)
    else:
        t_bin = t.copy()

    # Propensity score
    if z is not None:
        ps = _logistic_propensity_score(z, t_bin)
    else:
        ps = np.full(n, t_bin.mean())

    # Clip extreme propensities
    ps = np.clip(ps, 0.025, 0.975)

    # Weights
    p_treatment = t_bin.mean()
    if stabilize:
        w = t_bin * p_treatment / ps + (1 - t_bin) * (1 - p_treatment) / (1 - ps)
    else:
        w = t_bin / ps + (1 - t_bin) / (1 - ps)

    # Weighted means
    w_treated = w * t_bin
    w_control = w * (1 - t_bin)

    if w_treated.sum() < 1 or w_control.sum() < 1:
        return CausalEstimate(
            ate=0.0, std_error=np.inf, ci_lower=-np.inf, ci_upper=np.inf,
            method="IPW (failed)", n_samples=n,
            adjustment_set=adjustment_set or [],
            warnings=["Effective sample size too small; check propensity overlap"],
        )

    mu1 = np.average(y, weights=w_treated)
    mu0 = np.average(y, weights=w_control)
    ate = mu1 - mu0

    # Standard error via bootstrap-like approximation
    # Use the influence function approach (simplified)
    # SE = sqrt(var(IF) / n)
    if_z = (w_treated * (y - mu1)) / w_treated.sum() * n
    if_y0 = (w_control * (y - mu0)) / w_control.sum() * n
    if_ate = if_z - if_y0
    se_ate = np.sqrt(np.var(if_ate) / n)

    ci_lower = ate - 1.96 * se_ate
    ci_upper = ate + 1.96 * se_ate

    # Diagnostics
    max_weight = w.max()
    warnings = []
    if max_weight > 10:
        warnings.append(f"Large weights detected (max={max_weight:.1f}); "
                        "consider trimming or using stabilized weights")

    return CausalEstimate(
        ate=float(ate),
        std_error=float(se_ate),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        method="IPW" + (" (stabilized)" if stabilize else ""),
        n_samples=n,
        adjustment_set=adjustment_set or [],
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════
#  4. Doubly Robust Estimator
# ═══════════════════════════════════════════════════════════════════

def estimate_ate_doubly_robust(
    data: np.ndarray,
    var_names: List[str],
    treatment: str,
    outcome: str,
    adjustment_set: Optional[List[str]] = None,
) -> CausalEstimate:
    """
    Doubly Robust estimator (combines IPW + outcome regression).

    DR = (1/n) Σ [ μ₁(Z_i) - μ₀(Z_i)
                   + T_i(Y_i - μ₁(Z_i))/p_i
                   - (1-T_i)(Y_i - μ₀(Z_i))/(1-p_i) ]

    Consistent if EITHER the propensity model OR the outcome model
    is correctly specified. Hence "doubly" robust.
    """
    t, y, z = _get_columns(data, var_names, treatment, outcome, adjustment_set)
    n = len(t)

    # Binarize
    if len(np.unique(t)) > 2:
        t_median = np.median(t)
        t_bin = (t > t_median).astype(float)
    else:
        t_bin = t.copy()

    # Propensity score
    if z is not None:
        ps = np.clip(_logistic_propensity_score(z, t_bin), 0.025, 0.975)
    else:
        ps = np.full(n, np.clip(t_bin.mean(), 0.025, 0.975))

    # Outcome models: μ₁(Z) = E[Y | T=1, Z], μ₀(Z) = E[Y | T=0, Z]
    mask1 = t_bin == 1
    mask0 = t_bin == 0

    if z is not None:
        X1 = np.column_stack([np.ones(mask1.sum()), z[mask1]])
        X0 = np.column_stack([np.ones(mask0.sum()), z[mask0]])
        beta1, _, _, _ = np.linalg.lstsq(X1, y[mask1], rcond=None)
        beta0, _, _, _ = np.linalg.lstsq(X0, y[mask0], rcond=None)

        X_full = np.column_stack([np.ones(n), z])
        mu1 = X_full @ beta1
        mu0 = X_full @ beta0
    else:
        mu1 = np.full(n, y[mask1].mean() if mask1.sum() > 0 else 0)
        mu0 = np.full(n, y[mask0].mean() if mask0.sum() > 0 else 0)

    # DR estimator
    dr = (mu1 - mu0 +
          t_bin * (y - mu1) / ps -
          (1 - t_bin) * (y - mu0) / (1 - ps))
    ate = dr.mean()
    se_ate = dr.std() / np.sqrt(n)

    ci_lower = ate - 1.96 * se_ate
    ci_upper = ate + 1.96 * se_ate

    return CausalEstimate(
        ate=float(ate),
        std_error=float(se_ate),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        method="Doubly Robust",
        n_samples=n,
        adjustment_set=adjustment_set or [],
    )


# ═══════════════════════════════════════════════════════════════════
#  5. Stratification
# ═══════════════════════════════════════════════════════════════════

def estimate_ate_stratified(
    data: np.ndarray,
    var_names: List[str],
    treatment: str,
    outcome: str,
    adjustment_set: Optional[List[str]] = None,
    n_strata: int = 5,
) -> CausalEstimate:
    """
    Estimate ATE by stratifying on propensity score.

    1. Estimate propensity score
    2. Divide into n_strata quantiles
    3. Compute ATE within each stratum
    4. Weight strata by size
    """
    t, y, z = _get_columns(data, var_names, treatment, outcome, adjustment_set)
    n = len(t)

    if len(np.unique(t)) > 2:
        t_median = np.median(t)
        t_bin = (t > t_median).astype(float)
    else:
        t_bin = t.copy()

    if z is not None:
        ps = _logistic_propensity_score(z, t_bin)
    else:
        ps = np.full(n, t_bin.mean())

    # Stratify
    strata = pd_cut(ps, n_strata)
    ate_strata = []
    weights = []

    for s in range(n_strata):
        mask = strata == s
        n_s = mask.sum()
        if n_s < 10:
            continue

        t_s = t_bin[mask]
        y_s = y[mask]
        mask1 = t_s == 1
        mask0 = t_s == 0

        if mask1.sum() < 3 or mask0.sum() < 3:
            continue

        ate_s = y_s[mask1].mean() - y_s[mask0].mean()
        ate_strata.append(ate_s)
        weights.append(n_s)

    if not ate_strata:
        return CausalEstimate(
            ate=0.0, std_error=np.inf, ci_lower=-np.inf, ci_upper=np.inf,
            method="Stratification (failed)", n_samples=n,
            adjustment_set=adjustment_set or [],
            warnings=["Too few samples per stratum"],
        )

    weights = np.array(weights) / sum(weights)
    ate = np.average(ate_strata, weights=weights)

    # SE: weighted variance across strata
    var_ate = np.average((np.array(ate_strata) - ate) ** 2, weights=weights)
    se_ate = np.sqrt(var_ate / len(ate_strata))

    ci_lower = ate - 1.96 * se_ate
    ci_upper = ate + 1.96 * se_ate

    return CausalEstimate(
        ate=float(ate),
        std_error=float(se_ate),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        method=f"Stratification ({n_strata} strata)",
        n_samples=n,
        adjustment_set=adjustment_set or [],
    )


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _logistic_propensity_score(z: np.ndarray, t_bin: np.ndarray) -> np.ndarray:
    """Estimate P(T=1 | Z) via logistic regression (gradient descent)."""
    n, d = z.shape
    z_norm = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-8)
    X = np.column_stack([np.ones(n), z_norm])

    # Gradient descent
    w = np.zeros(d + 1)
    lr = 0.1
    for _ in range(200):
        logits = X @ w
        p = 1 / (1 + np.exp(-np.clip(logits, -10, 10)))
        grad = X.T @ (p - t_bin) / n
        w -= lr * grad

    ps = 1 / (1 + np.exp(-np.clip(X @ w, -10, 10)))
    return np.clip(ps, 0.01, 0.99)


def _smd(x1: np.ndarray, x0: np.ndarray) -> float:
    """Standardized Mean Difference between two groups."""
    s1, s0 = np.std(x1), np.std(x0)
    pooled_std = np.sqrt((s1**2 + s0**2) / 2)
    if pooled_std < 1e-12:
        return 0.0
    return abs(np.mean(x1) - np.mean(x0)) / pooled_std


def pd_cut(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign each value to a bin index (0..n_bins-1) by quantile."""
    q = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    q[0] = -np.inf
    q[-1] = np.inf
    bins = np.digitize(x, q[1:-1])
    return bins.clip(0, n_bins - 1)


# ═══════════════════════════════════════════════════════════════════
#  Main entry: auto-select best estimator
# ═══════════════════════════════════════════════════════════════════

def estimate_effect(
    data: np.ndarray,
    var_names: List[str],
    treatment: str,
    outcome: str,
    adjustment_set: Optional[List[str]] = None,
    method: str = "auto",
    **kwargs,
) -> CausalEstimate:
    """
    Estimate the ATE of treatment on outcome.

    Parameters
    ----------
    method : str
        'auto' → pick best based on data characteristics
        'linear', 'psm', 'ipw', 'dr', 'stratified'

    Returns
    -------
    CausalEstimate
    """
    if method == "linear":
        return estimate_ate_linear(data, var_names, treatment, outcome, adjustment_set)
    elif method == "psm":
        return estimate_ate_psm(data, var_names, treatment, outcome, adjustment_set, **kwargs)
    elif method == "ipw":
        return estimate_ate_ipw(data, var_names, treatment, outcome, adjustment_set, **kwargs)
    elif method == "dr" or method == "doubly_robust":
        return estimate_ate_doubly_robust(data, var_names, treatment, outcome, adjustment_set)
    elif method == "stratified":
        return estimate_ate_stratified(data, var_names, treatment, outcome, adjustment_set, **kwargs)
    elif method == "auto":
        # Default: linear for small adjustment sets, DR for larger ones
        n_adj = len(adjustment_set) if adjustment_set else 0
        if n_adj <= 2:
            return estimate_ate_linear(data, var_names, treatment, outcome, adjustment_set)
        else:
            return estimate_ate_doubly_robust(data, var_names, treatment, outcome, adjustment_set)
    else:
        raise ValueError(f"Unknown method: {method}. "
                         "Choose: linear, psm, ipw, dr, stratified, auto")


# ═══════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/duyw/causal_agent')

    print("=" * 60)
    print("  EFFECT ESTIMATION TESTS")
    print("=" * 60)

    # Generate synthetic data with known ATE
    from core.discovery import generate_linear_data
    from core.graph import CausalDAG
    from core.identification import identify_effect, find_back_door_adjustment

    rng = np.random.default_rng(42)

    # ── Test 1: Simple confounding (G→D, G→R, D→R) ──
    print("\n── Test 1: Simpson's Paradox (confounded) ──")
    dag = CausalDAG(["G", "D", "R"], [("G", "D"), ("G", "R"), ("D", "R")])
    data = generate_linear_data(dag, n_samples=2000, seed=123)
    var_names = ["G", "D", "R"]

    # Identify
    result = identify_effect(dag, "D", "R")
    adj = result.adjustment_set
    print(f"  Identification: {result.method}, adjust for {adj}")

    # True ATE (from data generation: D→R path coefficient is random)
    # For this test, we compare estimators against each other

    methods = ["linear", "psm", "ipw", "dr", "stratified"]
    for method in methods:
        est = estimate_effect(data, var_names, "D", "R", adj, method=method)
        sig = "✓ significant" if est.is_significant() else "✗ not significant"
        print(f"  {est.method:25s}: ATE={est.ate:+.4f}  "
              f"CI=[{est.ci_lower:+.3f}, {est.ci_upper:+.3f}]  {sig}")

    # ── Test 2: No confounding (X→M→Y) ──
    print("\n── Test 2: Chain X→M→Y (no confounding) ──")
    dag2 = CausalDAG(["X", "M", "Y"], [("X", "M"), ("M", "Y")])
    data2 = generate_linear_data(dag2, n_samples=1000, seed=456)
    var_names2 = ["X", "M", "Y"]

    for method in methods[:3]:  # just a few
        est = estimate_effect(data2, var_names2, "X", "Y", [], method=method)
        print(f"  {est.method:25s}: ATE={est.ate:+.4f}  "
              f"CI=[{est.ci_lower:+.3f}, {est.ci_upper:+.3f}]")

    # ── Test 3: Continuous treatment ──
    print("\n── Test 3: Continuous treatment ──")
    n_test = 1000
    Z = rng.normal(0, 1, n_test)
    T = 0.5 * Z + rng.normal(0, 0.5, n_test)  # continuous
    Y = 1.5 * T + 0.8 * Z + rng.normal(0, 0.3, n_test)

    data3 = np.column_stack([Z, T, Y])
    var_names3 = ["Z", "T", "Y"]

    for method in methods:
        est = estimate_effect(data3, var_names3, "T", "Y", ["Z"], method=method)
        bias = est.ate - 1.5  # true ATE = 1.5
        print(f"  {est.method:25s}: ATE={est.ate:+.4f}  "
              f"(true=1.5, bias={bias:+.4f})  {est.is_significant()}")

    # ── Test 4: Diagostics (balance check) ──
    print("\n── Test 4: Balance diagnostics ──")
    est_psm = estimate_effect(data, var_names, "D", "R", adj, method="psm")
    if est_psm.smd_before:
        print(f"  SMD before matching: {est_psm.smd_before}")
    if est_psm.smd_after:
        print(f"  SMD after matching:  {est_psm.smd_after}")
    if est_psm.propensity_overlap is not None:
        print(f"  Propensity overlap: {est_psm.propensity_overlap:.1%}")

    print("\n" + "=" * 60)
    print("  ALL ESTIMATION TESTS COMPLETE")
    print("=" * 60)

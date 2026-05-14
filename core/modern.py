"""
Modern Causal Inference Methods — Phase 3

  - Double Machine Learning (Chernozhukov et al., 2018)
  - CATE estimation: meta-learners + Causal Forest
  - do-why integration (optional backend)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .estimation import CausalEstimate


# ═══════════════════════════════════════════════════════════════════
#  Base ML models (lightweight, no external deps beyond numpy)
# ═══════════════════════════════════════════════════════════════════

class RidgeRegression:
    """Ridge regression with closed-form solution."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        X_aug = np.column_stack([np.ones(n), X])
        I = np.eye(d + 1)
        I[0, 0] = 0  # don't regularize intercept
        xtx = X_aug.T @ X_aug + self.alpha * I
        xty = X_aug.T @ y
        beta = np.linalg.solve(xtx, xty)
        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.intercept_ + X @ self.coef_


class PolynomialRidge:
    """Ridge regression with polynomial features (degree 2)."""

    def __init__(self, alpha: float = 1.0, degree: int = 2):
        self.alpha = alpha
        self.degree = degree
        self.model = RidgeRegression(alpha)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        feats = [X]
        if self.degree >= 2:
            # Add pairwise interactions
            for i in range(d):
                for j in range(i, d):
                    feats.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.column_stack(feats)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_poly = self._transform(X)
        self.model.fit(X_poly, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._transform(X))


def _default_ml_model():
    """Default ML model for nuisance functions."""
    return PolynomialRidge(alpha=0.1, degree=2)


# ═══════════════════════════════════════════════════════════════════
#  3.1 Double Machine Learning
# ═══════════════════════════════════════════════════════════════════

def estimate_ate_dml(
    data: np.ndarray,
    var_names: List[str],
    treatment: str,
    outcome: str,
    adjustment_set: Optional[List[str]] = None,
    n_folds: int = 5,
    model_y: Optional[object] = None,
    model_t: Optional[object] = None,
    seed: int = 42,
) -> CausalEstimate:
    """
    Double Machine Learning (DML) for ATE estimation.

    Algorithm (Chernozhukov et al., 2018):
      1. K-fold cross-fitting
      2. For each fold:
         a. Train E[Y|Z] and E[T|Z] on other folds
         b. Compute residuals: Y_res, T_res
         c. Orthogonal score: θ = (Y_res · T_res) / (T_res · T_res)
      3. ATE = mean(θ) across folds

    Advantages over linear regression:
      - No need to specify functional form of g(Z) or m(Z)
      - √n-consistent even with slow ML estimators
      - Valid inference (correct standard errors)
    """
    t, y, z = _extract(data, var_names, treatment, outcome, adjustment_set)
    n = len(t)
    rng = np.random.default_rng(seed)

    if model_y is None:
        model_y = _default_ml_model()
    if model_t is None:
        model_t = _default_ml_model()

    # Binarize treatment if continuous
    if len(np.unique(t)) > 10:
        t_bin = t.copy()  # keep continuous
    else:
        t_bin = t.copy()

    # Cross-fitting
    fold_idx = rng.permutation(n) % n_folds
    thetas = np.zeros(n_folds)

    for k in range(n_folds):
        train_mask = fold_idx != k
        test_mask = fold_idx == k

        # Nuisance functions on training data
        if z is not None:
            # Outcome model: E[Y|Z]
            g = RidgeRegression(alpha=0.5)
            g.fit(z[train_mask], y[train_mask])
            y_pred = g.predict(z)

            # Treatment model: E[T|Z]
            m = RidgeRegression(alpha=0.5)
            m.fit(z[train_mask], t_bin[train_mask])
            t_pred = m.predict(z)
        else:
            y_pred = np.full(n, y[train_mask].mean())
            t_pred = np.full(n, t_bin[train_mask].mean())

        # Residuals on test fold
        y_res = y[test_mask] - y_pred[test_mask]
        t_res = t_bin[test_mask] - t_pred[test_mask]

        # Orthogonal score
        if np.sum(t_res ** 2) < 1e-12:
            thetas[k] = 0.0
        else:
            thetas[k] = np.mean(y_res * t_res) / np.mean(t_res ** 2)

    ate = thetas.mean()
    se_ate = thetas.std() / np.sqrt(n_folds)

    ci_lower = ate - 1.96 * se_ate
    ci_upper = ate + 1.96 * se_ate

    warnings = []
    if n_folds < 5:
        warnings.append("Few folds — consider n_folds >= 5 for valid inference")

    return CausalEstimate(
        ate=float(ate),
        std_error=float(se_ate),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        method=f"DML (K={n_folds})",
        n_samples=n,
        adjustment_set=adjustment_set or [],
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════
#  3.2 Heterogeneous Treatment Effects (CATE)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CATEstimate:
    """Heterogeneous treatment effect estimate."""

    cate: np.ndarray          # individual treatment effects
    features: np.ndarray      # covariate matrix used
    ate: float                # average over all individuals
    method: str
    n_samples: int

    def summary(self, n_bins: int = 3) -> str:
        """Group-wise summary of CATE heterogeneity."""
        lines = [
            f"CATE Summary ({self.method})",
            f"  ATE = {self.ate:.4f}",
            f"  CATE range: [{self.cate.min():.4f}, {self.cate.max():.4f}]",
            "",
            "  Heterogeneity by quantile:",
        ]
        qs = np.percentile(self.cate, np.linspace(0, 100, n_bins + 1))
        for i in range(n_bins):
            mask = (self.cate >= qs[i]) & (self.cate < qs[i + 1])
            if mask.sum() > 0:
                lines.append(
                    f"    Q{i+1}: CATE={self.cate[mask].mean():+.4f}  "
                    f"(n={mask.sum()})"
                )
        return "\n".join(lines)


def estimate_cate_slearner(
    data: np.ndarray,
    var_names: List[str],
    treatment: str,
    outcome: str,
    covariates: List[str],
    model: Optional[object] = None,
) -> CATEstimate:
    """
    S-learner: Single model Y ~ f(T, X).

    CATE(x) = f(1, x) - f(0, x)
    """
    t, y, z = _extract(data, var_names, treatment, outcome, covariates)
    n = len(t)

    if len(np.unique(t)) > 2:
        t_bin = (t > np.median(t)).astype(float)
    else:
        t_bin = t

    if model is None:
        model = PolynomialRidge(alpha=0.1, degree=2)

    # Feature matrix: [T, X1, X2, ...]
    if z is not None:
        X = np.column_stack([t_bin, z])
    else:
        X = t_bin.reshape(-1, 1)

    model.fit(X, y)

    # CATE: difference in predictions
    X1 = X.copy()
    X1[:, 0] = 1.0
    X0 = X.copy()
    X0[:, 0] = 0.0

    cate = model.predict(X1) - model.predict(X0)

    return CATEstimate(
        cate=cate,
        features=X[:, 1:] if z is not None else np.zeros((n, 0)),
        ate=float(cate.mean()),
        method="S-learner",
        n_samples=n,
    )


def estimate_cate_tlearner(
    data: np.ndarray,
    var_names: List[str],
    treatment: str,
    outcome: str,
    covariates: List[str],
    model_t1: Optional[object] = None,
    model_t0: Optional[object] = None,
) -> CATEstimate:
    """
    T-learner: Two separate models.

    μ₁(X) = E[Y|T=1, X],  μ₀(X) = E[Y|T=0, X]
    CATE(x) = μ₁(x) - μ₀(x)
    """
    t, y, z = _extract(data, var_names, treatment, outcome, covariates)
    n = len(t)

    if len(np.unique(t)) > 2:
        t_bin = (t > np.median(t)).astype(float)
    else:
        t_bin = t

    if model_t1 is None:
        model_t1 = PolynomialRidge(alpha=0.1, degree=2)
    if model_t0 is None:
        model_t0 = PolynomialRidge(alpha=0.1, degree=2)

    if z is None:
        z = np.zeros((n, 0))

    mask1 = t_bin == 1
    mask0 = t_bin == 0

    model_t1.fit(z[mask1], y[mask1])
    model_t0.fit(z[mask0], y[mask0])

    mu1 = model_t1.predict(z)
    mu0 = model_t0.predict(z)
    cate = mu1 - mu0

    return CATEstimate(
        cate=cate,
        features=z,
        ate=float(cate.mean()),
        method="T-learner",
        n_samples=n,
    )


def estimate_cate_xlearner(
    data: np.ndarray,
    var_names: List[str],
    treatment: str,
    outcome: str,
    covariates: List[str],
    model_tau1: Optional[object] = None,
    model_tau0: Optional[object] = None,
) -> CATEstimate:
    """
    X-learner (Künzel et al., 2019).

    1. Train μ₁(X) on treated, μ₀(X) on control
    2. Impute counterfactuals:
       τ̃₁ = Y₁ - μ₀(X₁)  (treated units)
       τ̃₀ = μ₁(X₀) - Y₀  (control units)
    3. Train τ₁(X) on (X₁, τ̃₁), τ₀(X) on (X₀, τ̃₀)
    4. CATE(x) = propensity(x) · τ₀(x) + (1-prop(x)) · τ₁(x)

    Better than T-learner when one group is much smaller.
    """
    t, y, z = _extract(data, var_names, treatment, outcome, covariates)
    n = len(t)

    if len(np.unique(t)) > 2:
        t_bin = (t > np.median(t)).astype(float)
    else:
        t_bin = t

    if z is None:
        z = np.zeros((n, 0))

    mask1 = t_bin == 1
    mask0 = t_bin == 0

    # Step 1: two outcome models
    m1 = PolynomialRidge(alpha=0.1, degree=2)
    m0 = PolynomialRidge(alpha=0.1, degree=2)
    m1.fit(z[mask1], y[mask1])
    m0.fit(z[mask0], y[mask0])

    # Step 2: imputed treatment effects
    tau_tilde_1 = y[mask1] - m0.predict(z[mask1])  # treated: Y1 - μ₀(X)
    tau_tilde_0 = m1.predict(z[mask0]) - y[mask0]  # control: μ₁(X) - Y0

    # Step 3: CATE models
    if model_tau1 is None:
        model_tau1 = PolynomialRidge(alpha=0.1, degree=2)
    if model_tau0 is None:
        model_tau0 = PolynomialRidge(alpha=0.1, degree=2)

    model_tau1.fit(z[mask1], tau_tilde_1)
    model_tau0.fit(z[mask0], tau_tilde_0)

    # Step 4: propensity-weighted CATE
    # Simple propensity: P(T=1) for each unit
    prop_model = RidgeRegression(alpha=0.5)
    prop_model.fit(z, t_bin)
    prop = np.clip(prop_model.predict(z), 0.1, 0.9)

    tau1 = model_tau1.predict(z)
    tau0 = model_tau0.predict(z)

    cate = prop * tau0 + (1 - prop) * tau1

    return CATEstimate(
        cate=cate,
        features=z,
        ate=float(cate.mean()),
        method="X-learner",
        n_samples=n,
    )


# ═══════════════════════════════════════════════════════════════════
#  Simplified Causal Forest
# ═══════════════════════════════════════════════════════════════════

class SimpleCausalTree:
    """A single causal tree that estimates treatment effects in leaves."""

    def __init__(self, min_samples_leaf: int = 20, max_depth: int = 4):
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.split_var_: Optional[int] = None
        self.split_val_: float = 0.0
        self.left_: Optional[SimpleCausalTree] = None
        self.right_: Optional[SimpleCausalTree] = None
        self.ate_: float = 0.0  # leaf prediction

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray, depth: int = 0):
        n = len(y)
        self.ate_ = self._causal_mean(t, y)

        if (n < 2 * self.min_samples_leaf or depth >= self.max_depth or
                len(np.unique(t)) < 2):
            return self

        # Find best split
        best_gain = -np.inf
        best_var = -1
        best_val = 0.0

        for var in range(X.shape[1]):
            x_col = X[:, var]
            candidates = np.percentile(x_col, [25, 50, 75])

            for val in candidates:
                left_mask = x_col <= val
                right_mask = ~left_mask
                n_left = left_mask.sum()
                n_right = right_mask.sum()

                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                if len(np.unique(t[left_mask])) < 2 or len(np.unique(t[right_mask])) < 2:
                    continue

                ate_left = self._causal_mean(t[left_mask], y[left_mask])
                ate_right = self._causal_mean(t[right_mask], y[right_mask])

                # Gain = variance reduction in treatment effects
                gain = (n_left * (ate_left - self.ate_) ** 2 +
                        n_right * (ate_right - self.ate_) ** 2)

                if gain > best_gain:
                    best_gain = gain
                    best_var = var
                    best_val = val

        if best_var < 0:
            return self

        # Split
        left_mask = X[:, best_var] <= best_val
        right_mask = ~left_mask

        self.split_var_ = best_var
        self.split_val_ = best_val
        self.left_ = SimpleCausalTree(self.min_samples_leaf, self.max_depth)
        self.right_ = SimpleCausalTree(self.min_samples_leaf, self.max_depth)
        self.left_.fit(X[left_mask], t[left_mask], y[left_mask], depth + 1)
        self.right_.fit(X[right_mask], t[right_mask], y[right_mask], depth + 1)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.left_ is None:
            return np.full(len(X), self.ate_)

        preds = np.zeros(len(X))
        left_mask = X[:, self.split_var_] <= self.split_val_
        preds[left_mask] = self.left_.predict(X[left_mask])
        preds[~left_mask] = self.right_.predict(X[~left_mask])
        return preds

    @staticmethod
    def _causal_mean(t: np.ndarray, y: np.ndarray) -> float:
        mask1 = t > np.median(t) if len(np.unique(t)) > 2 else t == 1
        mask0 = ~mask1
        if mask1.sum() < 2 or mask0.sum() < 2:
            return 0.0
        return y[mask1].mean() - y[mask0].mean()


class SimpleCausalForest:
    """Ensemble of SimpleCausalTree for robust CATE estimation."""

    def __init__(self, n_trees: int = 20, min_samples_leaf: int = 30,
                 max_depth: int = 4, subsample_ratio: float = 0.5,
                 seed: int = 42):
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.subsample_ratio = subsample_ratio
        self.seed = seed
        self.trees: List[SimpleCausalTree] = []

    def fit(self, X: np.ndarray, t: np.ndarray, y: np.ndarray):
        rng = np.random.default_rng(self.seed)
        n = len(y)
        n_subsample = max(int(n * self.subsample_ratio), self.min_samples_leaf * 2)

        self.trees = []
        for i in range(self.n_trees):
            idx = rng.choice(n, n_subsample, replace=True)
            tree = SimpleCausalTree(self.min_samples_leaf, self.max_depth)
            tree.fit(X[idx], t[idx], y[idx])
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = np.zeros(len(X))
        for tree in self.trees:
            preds += tree.predict(X)
        return preds / len(self.trees)


def estimate_cate_forest(
    data: np.ndarray,
    var_names: List[str],
    treatment: str,
    outcome: str,
    covariates: List[str],
    n_trees: int = 30,
    max_depth: int = 4,
    seed: int = 42,
) -> CATEstimate:
    """Estimate heterogeneous treatment effects using a causal forest."""
    t, y, z = _extract(data, var_names, treatment, outcome, covariates)
    n = len(t)

    if len(np.unique(t)) > 2:
        t_bin = (t > np.median(t)).astype(float)
    else:
        t_bin = t

    if z is None:
        return CATEstimate(
            cate=np.full(n, y[t_bin == 1].mean() - y[t_bin == 0].mean()),
            features=np.zeros((n, 0)),
            ate=float(y[t_bin == 1].mean() - y[t_bin == 0].mean()),
            method="Causal Forest (constant)",
            n_samples=n,
        )

    forest = SimpleCausalForest(
        n_trees=n_trees, min_samples_leaf=30,
        max_depth=max_depth, seed=seed,
    )
    forest.fit(z, t_bin, y)
    cate = forest.predict(z)

    return CATEstimate(
        cate=cate,
        features=z,
        ate=float(cate.mean()),
        method=f"Causal Forest ({n_trees} trees)",
        n_samples=n,
    )


# ═══════════════════════════════════════════════════════════════════
#  3.3 do-why Integration (optional backend)
# ═══════════════════════════════════════════════════════════════════

def estimate_ate_dowhy(
    data: np.ndarray,
    var_names: List[str],
    treatment: str,
    outcome: str,
    adjustment_set: Optional[List[str]] = None,
    method: str = "backdoor.linear_regression",
) -> Optional[CausalEstimate]:
    """
    Estimate ATE using do-why (if installed).

    Install: pip install dowhy

    Returns None if do-why is not available.
    """
    try:
        import dowhy
        from dowhy import CausalModel
        import pandas as pd
    except ImportError:
        return None

    df = pd.DataFrame(data, columns=var_names)
    dag_spec = ""
    # Build DAG specification from edges
    # (we'd need the DAG, which isn't passed here — simplified)

    try:
        model = CausalModel(
            data=df,
            treatment=treatment,
            outcome=outcome,
            common_causes=adjustment_set or [],
        )
        identified = model.identify_effect()
        estimate = model.estimate_effect(
            identified,
            method_name=method,
        )
        return CausalEstimate(
            ate=float(estimate.value),
            std_error=float(estimate.get_standard_error() or 0.0),
            ci_lower=float(estimate.get_confidence_intervals()[0]
                           if estimate.get_confidence_intervals() else 0),
            ci_upper=float(estimate.get_confidence_intervals()[1]
                           if estimate.get_confidence_intervals() else 0),
            method=f"do-why ({method})",
            n_samples=len(data),
            adjustment_set=adjustment_set or [],
        )
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _extract(
    data: np.ndarray, var_names: List[str],
    treatment: str, outcome: str,
    covariates: Optional[List[str]],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extract treatment, outcome, and covariate columns."""
    name_to_idx = {v: i for i, v in enumerate(var_names)}
    t = data[:, name_to_idx[treatment]]
    y = data[:, name_to_idx[outcome]]
    if covariates:
        z = data[:, [name_to_idx[c] for c in covariates]]
    else:
        z = None
    return t, y, z


# ═══════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/duyw/causal_agent')
    from core.discovery import generate_linear_data
    from core.graph import CausalDAG

    print("=" * 55)
    print("  PHASE 3 — MODERN METHODS TESTS")
    print("=" * 55)

    # Generate data
    dag = CausalDAG(["G", "D", "R"], [("G", "D"), ("G", "R"), ("D", "R")])
    data = generate_linear_data(dag, 3000, seed=42)
    var_names = ["G", "D", "R"]

    # Test DML
    print("\n── 3.1 Double ML ──")
    est_dml = estimate_ate_dml(data, var_names, "D", "R", ["G"], n_folds=5)
    print(est_dml.summary())
    print(f"  Significant: {est_dml.is_significant()}")

    # Test S-learner
    print("\n── 3.2 S-learner CATE ──")
    cate_s = estimate_cate_slearner(data, var_names, "D", "R", ["G"])
    print(cate_s.summary())

    # Test T-learner
    print("\n── 3.3 T-learner CATE ──")
    cate_t = estimate_cate_tlearner(data, var_names, "D", "R", ["G"])
    print(cate_t.summary())

    # Test X-learner
    print("\n── 3.4 X-learner CATE ──")
    cate_x = estimate_cate_xlearner(data, var_names, "D", "R", ["G"])
    print(cate_x.summary())

    # Test Causal Forest
    print("\n── 3.5 Causal Forest ──")
    cate_f = estimate_cate_forest(data, var_names, "D", "R", ["G"], n_trees=20)
    print(cate_f.summary())

    # Compare all methods
    print("\n── Comparison ──")
    methods = [
        ("DML", est_dml),
        ("S-learner", cate_s),
        ("T-learner", cate_t),
        ("X-learner", cate_x),
        ("Causal Forest", cate_f),
    ]
    for name, est in methods:
        ate = est.ate if hasattr(est, 'ate') else est.ate
        print(f"  {name:20s}: ATE = {ate:+.4f}")

    print("\n" + "=" * 55)
    print("  ALL PHASE 3 TESTS COMPLETE")
    print("=" * 55)

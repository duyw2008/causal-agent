"""
Time Series Causal Discovery — 时间序列因果发现

Algorithms:
  - Granger Causality (VAR-based F-test)
  - Time-Series PC (temporal ordering constraint)
  - PCMCI-lite (simplified version of Runge et al., 2019)

Key insight: time order provides a natural causal constraint —
  causes must precede effects in time. This breaks the Markov
  equivalence class ambiguity that plagues i.i.d. causal discovery.

Reference: Runge, J. et al. (2019). "Detecting and quantifying
  causal associations in large nonlinear time series datasets."
  Science Advances.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .graph import CausalDAG
from .discovery import fisher_z_test


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _lagged_data(
    data: np.ndarray,
    max_lag: int,
    var_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert time series to lagged design matrix.

    Input:  data[n_timepoints, n_vars]
    Output: X[n_timepoints - max_lag, n_vars * (max_lag + 1)]

    Columns: [X1(t), X2(t), ..., X1(t-1), X2(t-1), ..., X1(t-max_lag), ...]
    """
    n_time, n_vars = data.shape
    if var_names is None:
        var_names = [f"V{i}" for i in range(n_vars)]

    n_effective = n_time - max_lag
    X = np.zeros((n_effective, n_vars * (max_lag + 1)))
    col_names = []

    # Current values
    for v in range(n_vars):
        X[:, v] = data[max_lag:, v]
        col_names.append(f"{var_names[v]}(t)")

    # Lagged values
    for lag in range(1, max_lag + 1):
        offset = n_vars * lag
        for v in range(n_vars):
            X[:, offset + v] = data[max_lag - lag:n_time - lag, v]
            col_names.append(f"{var_names[v]}(t-{lag})")

    return X, col_names


def _var_residuals(
    data: np.ndarray,
    max_lag: int,
    target_idx: int,
    exclude_idxs: Optional[Set[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute residuals of VAR model for Granger test.

    Returns (residuals_restricted, residuals_full) where:
      restricted: Y ~ past(Y) + past(other_controls)
      full:       Y ~ past(Y) + past(X) + past(other_controls)
    """
    n_time, n_vars = data.shape
    if exclude_idxs is None:
        exclude_idxs = set()

    # Build lagged design matrix
    X_design, _ = _lagged_data(data, max_lag, [f"V{i}" for i in range(n_vars)])
    Y_target = X_design[:, target_idx]  # Y(t)

    # Restricted model: only past of Y + other controls (no X)
    restricted_cols = []
    for lag in range(1, max_lag + 1):
        restricted_cols.append(n_vars * lag + target_idx)
    for v in range(n_vars):
        if v != target_idx and v not in exclude_idxs:
            for lag in range(1, max_lag + 1):
                restricted_cols.append(n_vars * lag + v)

    X_restricted = X_design[:, restricted_cols]
    beta_r, _, _, _ = np.linalg.lstsq(
        np.column_stack([np.ones(len(X_restricted)), X_restricted]),
        Y_target, rcond=None
    )
    resid_r = Y_target - np.column_stack([np.ones(len(X_restricted)), X_restricted]) @ beta_r

    # Full model: add past of X
    full_cols = list(restricted_cols)
    for xi in exclude_idxs:
        for lag in range(1, max_lag + 1):
            full_cols.append(n_vars * lag + xi)

    X_full = X_design[:, full_cols]
    beta_f, _, _, _ = np.linalg.lstsq(
        np.column_stack([np.ones(len(X_full)), X_full]),
        Y_target, rcond=None
    )
    resid_f = Y_target - np.column_stack([np.ones(len(X_full)), X_full]) @ beta_f

    return resid_r, resid_f


# ═══════════════════════════════════════════════════════════════════
#  Granger Causality
# ═══════════════════════════════════════════════════════════════════

def granger_causality_test(
    data: np.ndarray,
    var_names: List[str],
    max_lag: int = 3,
    alpha: float = 0.05,
) -> Dict[str, List[Tuple[str, float, bool]]]:
    """
    Pairwise Granger causality test.

    H₀: X does NOT Granger-cause Y
    (Past values of X do not help predict Y beyond past values of Y)

    For each pair (X, Y):
      1. Restricted:  Y(t) ~ past(Y) + past(all_others)
      2. Full:        Y(t) ~ past(Y) + past(X) + past(all_others)
      3. F-test:      (SSR_r - SSR_f) / SSR_f  ~  F(max_lag, n - k)

    Returns dict: {cause_var: [(effect_var, p_value, is_significant), ...]}
    """
    n_time, n_vars = data.shape
    n_eff = n_time - max_lag

    results: Dict[str, List[Tuple[str, float, bool]]] = {v: [] for v in var_names}

    for x_idx in range(n_vars):
        for y_idx in range(n_vars):
            if x_idx == y_idx:
                continue

            # Compute residuals
            resid_r, resid_f = _var_residuals(
                data, max_lag, y_idx, exclude_idxs={x_idx}
            )

            ssr_r = np.sum(resid_r ** 2)
            ssr_f = np.sum(resid_f ** 2)

            if ssr_f < 1e-12:
                p_value = 1.0
            else:
                # F-statistic
                # Additional params = max_lag (one per lag of X)
                n_restricted_params = len(resid_r) - np.sum(resid_r ** 2) / max(ssr_r, 1e-12)
                f_stat = ((ssr_r - ssr_f) / max_lag) / (ssr_f / (n_eff - n_vars * max_lag - 1))
                f_stat = max(f_stat, 0)

                # Approximate p-value using F-distribution
                # (simplified: use chi-squared approximation for large n)
                p_value = np.exp(-f_stat * max_lag / 2)

            is_sig = p_value < alpha
            results[var_names[x_idx]].append(
                (var_names[y_idx], float(p_value), is_sig)
            )

    return results


def granger_discover_dag(
    data: np.ndarray,
    var_names: List[str],
    max_lag: int = 3,
    alpha: float = 0.05,
) -> CausalDAG:
    """
    Discover causal DAG using pairwise Granger tests.

    Edge X→Y exists if X Granger-causes Y at significance alpha.

    Returns a CausalDAG with contemporaneous + lagged edges.
    """
    results = granger_causality_test(data, var_names, max_lag, alpha)
    edges = []
    for cause, effects in results.items():
        for effect, p_val, is_sig in effects:
            if is_sig:
                edges.append((cause, effect))

    # Remove cycles (if any — shouldn't happen with Granger)
    # by only keeping edges consistent with time ordering
    # (This is automatically satisfied since Granger tests temporal precedence)

    return CausalDAG(var_names, edges)


# ═══════════════════════════════════════════════════════════════════
#  Time-Series PC (Temporal Ordering Constraint)
# ═══════════════════════════════════════════════════════════════════

def ts_pc_algorithm(
    data: np.ndarray,
    var_names: Optional[List[str]] = None,
    max_lag: int = 3,
    alpha: float = 0.05,
    max_cond_size: int = 5,
    verbose: bool = False,
) -> CausalDAG:
    """
    PC algorithm adapted for time series.

    Key difference from i.i.d. PC:
      - Variables at time t can only be caused by variables at time s ≤ t
      - This eliminates the Markov equivalence ambiguity for lagged edges
      - Lagged edges are always directed: past → present

    We create a "time-unfolded" graph with nodes X_i(t-lag) and test
    conditional independencies with the temporal constraint baked in.
    """
    n_time, n_vars = data.shape
    if var_names is None:
        var_names = [f"V{i}" for i in range(n_vars)]

    # Build lagged design matrix
    X_design, col_names = _lagged_data(data, max_lag, var_names)
    n_eff = len(X_design)

    # Total variables in unfolded graph = n_vars * (max_lag + 1)
    n_total = len(col_names)

    # Only test edges FROM lagged TO contemporaneous
    # (and contemporaneous → contemporaneous)
    contemp_indices = list(range(n_vars))  # X_i(t)

    # Initialize adjacency: only allow edges from past to present/future
    adjacency: Dict[int, Set[int]] = {i: set() for i in range(n_total)}

    # Add candidate edges:
    # - Past → Present (always allowed)
    # - Present → Present (bi-directional candidates)
    for i in range(n_total):
        for j in range(n_total):
            if i == j:
                continue
            # Determine if i is "before" j in time
            i_lag = _get_lag(i, n_vars)
            j_lag = _get_lag(j, n_vars)
            if i_lag > j_lag:
                # i is further in the past → i can cause j
                adjacency[i].add(j)
            elif i_lag == j_lag and i_lag == 0:
                # Both contemporaneous → add bidirectional candidate
                adjacency[i].add(j)

    if verbose:
        n_candidates = sum(len(adjacency[i]) for i in range(n_total)) // 2
        print(f"[TS-PC] {n_total} unfolded variables, ~{n_candidates} candidate edges")

    # Conditional independence tests (only on allowed edges)
    cond_size = 0
    while cond_size <= max_cond_size:
        removed_any = False
        for i in range(n_total):
            for j in sorted(adjacency[i]):
                if j <= i:
                    continue
                candidates = [n for n in adjacency[i] if n != j]
                if len(candidates) < cond_size:
                    continue
                # Test subsets
                from itertools import combinations
                found_sep = False
                for S_idx in combinations(candidates, cond_size):
                    S = list(S_idx)
                    indep, _ = fisher_z_test(X_design, i, j, S, alpha)
                    if indep:
                        adjacency[i].discard(j)
                        adjacency[j].discard(i)
                        if verbose:
                            print(f"  Removed {col_names[i]} — {col_names[j]} "
                                  f"| {{{', '.join(col_names[s] for s in S)}}}")
                        found_sep = True
                        removed_any = True
                        break
                if found_sep:
                    break
        cond_size += 1
        if not removed_any:
            break

    # Convert to CausalDAG
    # Only keep contemporaneous edges for the final DAG
    # (lagged edges are implicit in the time structure)
    edges = []
    for i in contemp_indices:
        for j in adjacency[i]:
            if j in contemp_indices and i < j:
                edges.append((var_names[i], var_names[j]))

    # Add lagged edges as "lagged_self" annotations
    # (simplified: only return contemporaneous DAG)
    return CausalDAG(var_names, edges)


def _get_lag(idx: int, n_vars: int) -> int:
    """Get the time lag of an unfolded variable index."""
    return idx // n_vars


# ═══════════════════════════════════════════════════════════════════
#  PCMCI-lite (simplified)
# ═══════════════════════════════════════════════════════════════════

def pcmci_lite(
    data: np.ndarray,
    var_names: Optional[List[str]] = None,
    max_lag: int = 3,
    alpha: float = 0.05,
    verbose: bool = False,
) -> Dict[str, List[Tuple[str, int, float]]]:
    """
    Simplified PCMCI algorithm (Runge et al., 2019).

    PCMCI = PC (skeleton) + MCI (Momentary Conditional Independence)

    Steps:
      1. PC₁: Run PC on time-unfolded graph to get preliminary parents
      2. MCI: For each pair (X_{t-τ}, Y_t), test:
           X_{t-τ} ⊥ Y_t | Parents(Y_t) \\ {X_{t-τ}} ∪ Parents(X_{t-τ})

    This removes false positives that arise from auto-correlation.

    Returns: {cause_var: [(effect_var, lag, p_value), ...]}
    """
    n_time, n_vars = data.shape
    if var_names is None:
        var_names = [f"V{i}" for i in range(n_vars)]

    X_design, col_names = _lagged_data(data, max_lag, var_names)
    n_total = len(col_names)
    contemp_indices = list(range(n_vars))

    # ── Step 1: PC skeleton on unfolded graph ─────────────────
    adjacency: Dict[int, Set[int]] = {i: set() for i in range(n_total)}
    for i in range(n_total):
        for j in range(n_total):
            if i == j:
                continue
            i_lag = _get_lag(i, n_vars)
            j_lag = _get_lag(j, n_vars)
            if i_lag > j_lag:
                adjacency[i].add(j)  # past → present
            elif i_lag == j_lag and i_lag == 0:
                adjacency[i].add(j)  # contemporaneous

    # Quick PC (simplified: only test unconditional + first-order)
    for size in range(min(3, n_total)):
        for i in range(n_total):
            for j in sorted(adjacency[i]):
                if j <= i:
                    continue
                cand = [n for n in adjacency[i] if n != j]
                if len(cand) < size:
                    continue
                from itertools import combinations
                for S in combinations(cand, size):
                    indep, _ = fisher_z_test(X_design, i, j, list(S), alpha)
                    if indep:
                        adjacency[i].discard(j)
                        adjacency[j].discard(i)
                        break

    # ── Step 2: MCI — test lagged → contemporaneous ───────────
    results: Dict[str, List[Tuple[str, int, float]]] = {v: [] for v in var_names}

    for y_idx in contemp_indices:
        y_name = var_names[y_idx]
        parents_y = list(adjacency[y_idx])  # preliminary parents of Y(t)

        for x_lag_idx in range(n_vars, n_total):  # only lagged variables
            if x_lag_idx not in adjacency[y_idx]:
                continue
            x_lag = _get_lag(x_lag_idx, n_vars)
            x_var_idx = x_lag_idx % n_vars
            x_name = var_names[x_var_idx]

            # MCI conditioning set:
            # Parents(Y_t) \ {X_{t-τ}}  ∪  Parents(X_{t-τ})
            cond_set = set(parents_y) - {x_lag_idx}
            # Add parents of X at this lag (simplified: use all lagged parents)
            cond_set |= set(adjacency.get(x_lag_idx, set()))

            cond_list = list(cond_set - {y_idx, x_lag_idx})
            if len(cond_list) > 10:
                cond_list = cond_list[:10]  # cap for performance

            indep, p_val = fisher_z_test(X_design, x_lag_idx, y_idx, cond_list, alpha)

            if not indep:
                results[x_name].append((y_name, x_lag, float(p_val)))

    return results


def pcmci_discover_dag(
    data: np.ndarray,
    var_names: Optional[List[str]] = None,
    max_lag: int = 3,
    alpha: float = 0.05,
    verbose: bool = False,
) -> CausalDAG:
    """
    Discover contemporaneous causal DAG using PCMCI.

    Lagged edges are included as annotations but the returned DAG
    focuses on contemporaneous causal structure.
    """
    if var_names is None:
        var_names = [f"V{i}" for i in range(data.shape[1])]

    pcmci_results = pcmci_lite(data, var_names, max_lag, alpha, verbose)

    # Build contemporaneous edges from PCMCI results
    edges = []
    for cause, effects in pcmci_results.items():
        for effect, lag, p_val in effects:
            if lag == 0:  # contemporaneous
                edges.append((cause, effect))
            # lagged edges are implicit in the time structure

    return CausalDAG(var_names, edges)


# ═══════════════════════════════════════════════════════════════════
#  Summary Report
# ═══════════════════════════════════════════════════════════════════

def ts_discovery_report(
    data: np.ndarray,
    var_names: List[str],
    max_lag: int = 3,
    alpha: float = 0.05,
) -> str:
    """Generate a comprehensive time series causal discovery report."""
    lines = [
        "=" * 60,
        "  TIME SERIES CAUSAL DISCOVERY REPORT",
        "=" * 60,
        f"  Variables: {', '.join(var_names)}",
        f"  Max lag: {max_lag}",
        f"  Timepoints: {data.shape[0]}",
        "",
    ]

    # Granger
    lines.append("── Granger Causality ──")
    gc = granger_causality_test(data, var_names, max_lag, alpha)
    for cause, effects in gc.items():
        sig_effects = [(e, p) for e, p, s in effects if s]
        if sig_effects:
            for effect, p_val in sig_effects:
                lines.append(f"  {cause} → {effect}  (p={p_val:.4f})")

    if not any(any(s for _, _, s in eff) for eff in gc.values()):
        lines.append("  (no significant Granger-causal edges found)")

    # PCMCI
    lines.append("")
    lines.append("── PCMCI ──")
    pcmci = pcmci_lite(data, var_names, max_lag, alpha)
    for cause, effects in pcmci.items():
        if effects:
            for effect, lag, p_val in effects:
                lag_str = f"(t-{lag})" if lag > 0 else "(t)"
                lines.append(f"  {cause}{lag_str} → {effect}  (p={p_val:.4f})")

    # Comparison
    lines.append("")
    lines.append("── Comparison ──")
    gc_dag = granger_discover_dag(data, var_names, max_lag, alpha)
    pcmci_dag = pcmci_discover_dag(data, var_names, max_lag, alpha)
    lines.append(f"  Granger DAG: {gc_dag}")
    lines.append(f"  PCMCI DAG:   {pcmci_dag}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Test
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  TIME SERIES CAUSAL DISCOVERY — TESTS")
    print("=" * 60)

    # ── Test 1: Simple VAR(1) process ──
    print("\n── Test 1: VAR(1) — X→Y ──")
    rng = np.random.default_rng(42)
    n_time = 500
    X = np.zeros(n_time)
    Y = np.zeros(n_time)
    for t in range(1, n_time):
        X[t] = 0.7 * X[t-1] + rng.normal(0, 0.3)
        Y[t] = 0.5 * Y[t-1] + 0.6 * X[t-1] + rng.normal(0, 0.3)
    data = np.column_stack([X, Y])

    gc = granger_causality_test(data, ["X", "Y"], max_lag=2, alpha=0.01)
    for cause, effects in gc.items():
        for effect, p_val, sig in effects:
            status = "✓ SIG" if sig else "— not sig"
            print(f"  {cause} → {effect}: p={p_val:.6f}  {status}")

    # ── Test 2: Bidirectional (X↔Y) ──
    print("\n── Test 2: Bidirectional feedback ──")
    n_time = 500
    X2 = np.zeros(n_time)
    Y2 = np.zeros(n_time)
    for t in range(1, n_time):
        X2[t] = 0.6 * X2[t-1] + 0.3 * Y2[t-1] + rng.normal(0, 0.3)
        Y2[t] = 0.6 * Y2[t-1] + 0.3 * X2[t-1] + rng.normal(0, 0.3)
    data2 = np.column_stack([X2, Y2])

    gc2 = granger_causality_test(data2, ["X", "Y"], max_lag=2, alpha=0.01)
    for cause, effects in gc2.items():
        for effect, p_val, sig in effects:
            status = "✓ SIG" if sig else "— not sig"
            print(f"  {cause} → {effect}: p={p_val:.6f}  {status}")

    # ── Test 3: 3-variable chain ──
    print("\n── Test 3: 3-var chain X→Y→Z ──")
    n_time = 500
    X3 = np.zeros(n_time); Y3 = np.zeros(n_time); Z3 = np.zeros(n_time)
    for t in range(1, n_time):
        X3[t] = 0.7 * X3[t-1] + rng.normal(0, 0.3)
        Y3[t] = 0.5 * Y3[t-1] + 0.5 * X3[t-1] + rng.normal(0, 0.3)
        Z3[t] = 0.5 * Z3[t-1] + 0.5 * Y3[t-1] + rng.normal(0, 0.3)
    data3 = np.column_stack([X3, Y3, Z3])

    print(ts_discovery_report(data3, ["X", "Y", "Z"], max_lag=2, alpha=0.01))

    # ── Test 4: PCMCI vs Granger ──
    print("\n── Test 4: PCMCI vs Granger ──")
    gc_dag = granger_discover_dag(data3, ["X", "Y", "Z"], max_lag=2, alpha=0.01)
    pcmci_dag = pcmci_discover_dag(data3, ["X", "Y", "Z"], max_lag=2, alpha=0.01)
    print(f"  Granger: {gc_dag}")
    print(f"  PCMCI:   {pcmci_dag}")

    print(f"\n{'='*60}")
    print("  TIME SERIES DISCOVERY TESTS COMPLETE")
    print(f"{'='*60}")

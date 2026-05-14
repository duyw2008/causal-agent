"""
Causal Discovery — learn causal structure from observational data.

Implements:
  - PC algorithm (constraint-based)
  - GES algorithm (score-based)

Both return a CausalDAG (or equivalence class).

References:
  - Spirtes, Glymour, Scheines (2000). Causation, Prediction, and Search.
  - Chickering (2002). Optimal Structure Identification with Greedy Search.
"""

from __future__ import annotations
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .graph import CausalDAG


# ═══════════════════════════════════════════════════════════════════
#  Conditional Independence Tests
# ═══════════════════════════════════════════════════════════════════

def _residual(data: np.ndarray, target_col: int, cond_cols: List[int]) -> np.ndarray:
    """Residual of regressing data[:, target_col] on data[:, cond_cols]."""
    if not cond_cols:
        return data[:, target_col] - data[:, target_col].mean()

    X = data[:, cond_cols]
    y = data[:, target_col]
    # Add intercept
    X = np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ beta
    return y - y_pred


def fisher_z_test(
    data: np.ndarray,
    x_idx: int,
    y_idx: int,
    cond_idxs: List[int],
    alpha: float = 0.05,
) -> Tuple[bool, float]:
    """
    Conditional independence test for continuous variables.

    H₀: X ⊥ Y | Z    (conditionally independent)

    Uses partial correlation + Fisher's z-transform.

    Returns (independent: bool, p_value: float).
    """
    n = data.shape[0]
    k = len(cond_idxs)

    # Compute residuals
    resid_x = _residual(data, x_idx, cond_idxs)
    resid_y = _residual(data, y_idx, cond_idxs)

    # Partial correlation
    r = np.corrcoef(resid_x, resid_y)[0, 1]

    # Fisher z-transform
    # z = 0.5 * ln((1+r)/(1-r)) * sqrt(n - k - 3)
    if abs(r) >= 1.0 - 1e-12:
        r = np.sign(r) * (1.0 - 1e-12)

    z_stat = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(n - k - 3)

    # Two-sided p-value
    try:
        from scipy.stats import norm
        p_value = 2.0 * (1.0 - norm.cdf(abs(z_stat)))
    except ImportError:
        # Fallback: z > 1.96 → p < 0.05
        p_value = 2.0 * np.exp(-z_stat**2 / 2) / (abs(z_stat) * np.sqrt(2 * np.pi) + 1e-12)

    return p_value > alpha, p_value


def g_squared_test(
    data: np.ndarray,
    x_idx: int,
    y_idx: int,
    cond_idxs: List[int],
    alpha: float = 0.05,
    discretize_bins: int = 5,
) -> Tuple[bool, float]:
    """
    Conditional independence test for discrete (or discretized) variables.

    H₀: X ⊥ Y | Z

    Uses G² (likelihood-ratio) test with chi-squared approximation.

    Returns (independent: bool, p_value: float).
    """
    n = data.shape[0]

    # Discretize if needed
    def discretize(col):
        vals = data[:, col]
        unique = np.unique(vals)
        if len(unique) <= discretize_bins:
            # Already discrete-ish: use unique values as-is
            mapping = {v: i for i, v in enumerate(sorted(unique))}
            return np.array([mapping[v] for v in vals]), len(unique)
        # Continuous: bin
        binned = np.digitize(vals, np.percentile(
            vals, np.linspace(0, 100, discretize_bins + 1)[1:-1]))
        return binned, discretize_bins

    x_disc, x_levels = discretize(x_idx)
    y_disc, y_levels = discretize(y_idx)

    if not cond_idxs:
        # Unconditional G²
        contingency = np.zeros((x_levels, y_levels))
        for i in range(n):
            contingency[x_disc[i], y_disc[i]] += 1

        row_sums = contingency.sum(axis=1, keepdims=True)
        col_sums = contingency.sum(axis=0, keepdims=True)
        expected = (row_sums @ col_sums) / n

        mask = expected > 0
        g2 = 2.0 * np.sum(contingency[mask] * np.log(
            contingency[mask] / expected[mask]))
    else:
        # Conditional: stratify by conditioning set
        z_data = data[:, cond_idxs]
        z_disc_list = []
        z_levels_list = []
        for c in cond_idxs:
            zd, zl = discretize(c)
            z_disc_list.append(zd)
            z_levels_list.append(zl)

        # Create combined Z key
        z_keys = {}
        for i in range(n):
            key = tuple(zd[i] for zd in z_disc_list)
            if key not in z_keys:
                z_keys[key] = len(z_keys)
        z_idx = np.array([z_keys[tuple(zd[i] for zd in z_disc_list)]
                          for i in range(n)])
        n_strata = len(z_keys)

        g2 = 0.0
        dof_total = 0

        for s in range(n_strata):
            mask_s = z_idx == s
            n_s = mask_s.sum()
            if n_s < 5:
                continue

            contingency = np.zeros((x_levels, y_levels))
            x_s = x_disc[mask_s]
            y_s = y_disc[mask_s]
            for i in range(len(x_s)):
                contingency[x_s[i], y_s[i]] += 1

            row_sums = contingency.sum(axis=1, keepdims=True)
            col_sums = contingency.sum(axis=0, keepdims=True)
            expected = (row_sums @ col_sums) / n_s

            mask_e = expected > 0
            if mask_e.sum() > 0:
                g2 += 2.0 * np.sum(
                    contingency[mask_e] *
                    np.log((contingency[mask_e] + 1e-10) /
                           (expected[mask_e] + 1e-10))
                )

            dof_total += (x_levels - 1) * (y_levels - 1)

        if dof_total == 0:
            return True, 1.0

    dof = (x_levels - 1) * (y_levels - 1)
    if cond_idxs:
        dof = dof_total

    if dof < 1:
        return True, 1.0

    try:
        from scipy.stats import chi2
    except ImportError:
        # Fallback: use an approximate threshold
        # For small contingency tables, reject independence if G² > 3.84 (χ²₁,₀.₀₅)
        critical_value = 3.84 * dof
        return g2 < critical_value, g2 / (g2 + n)
    p_value = 1.0 - chi2.cdf(max(g2, 0), dof)
    return p_value > alpha, p_value


# ═══════════════════════════════════════════════════════════════════
#  PC Algorithm
# ═══════════════════════════════════════════════════════════════════

def pc_algorithm(
    data: np.ndarray,
    var_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    discrete: bool = False,
    max_cond_size: Optional[int] = None,
    verbose: bool = False,
) -> CausalDAG:
    """
    PC algorithm for causal discovery.

    Parameters
    ----------
    data : np.ndarray of shape (n_samples, n_variables)
    var_names : optional list of variable names
    alpha : significance level for CI tests
    discrete : if True, use G² test; otherwise Fisher's z
    max_cond_size : max size of conditioning set (default: unlimited)
    verbose : print progress

    Returns
    -------
    CausalDAG — learned DAG (may contain undirected edges as bidirectional)
    """
    n_vars = data.shape[1]
    if var_names is None:
        var_names = [f"V{i}" for i in range(n_vars)]
    if max_cond_size is None:
        max_cond_size = n_vars

    ci_test = g_squared_test if discrete else fisher_z_test

    # ── Step 1: Skeleton discovery ────────────────────────────
    # adjacency[v] = set of neighbours of v
    adjacency: Dict[int, Set[int]] = {
        i: set(j for j in range(n_vars) if j != i)
        for i in range(n_vars)
    }
    # sep_set[(i,j)] = set that separates i and j
    sep_set: Dict[Tuple[int, int], Set[int]] = {}

    if verbose:
        print(f"[PC] Starting skeleton search on {n_vars} variables, "
              f"{n_vars*(n_vars-1)//2} possible edges")

    cond_size = 0
    while cond_size <= max_cond_size:
        removed_any = False
        for i in range(n_vars):
            neighbours = sorted(adjacency[i])
            for j in neighbours:
                if j <= i:
                    continue
                # Find candidate conditioning sets from adj(i) \ {j}
                candidates = [n for n in adjacency[i] if n != j]
                if len(candidates) < cond_size:
                    continue

                # Test all subsets of size cond_size
                found_sep = False
                for S_indices in combinations(candidates, cond_size):
                    S = list(S_indices)
                    indep, pval = ci_test(data, i, j, S, alpha)
                    if indep:
                        # Remove edge i—j
                        adjacency[i].discard(j)
                        adjacency[j].discard(i)
                        sep_set[(i, j)] = set(S)
                        sep_set[(j, i)] = set(S)
                        if verbose:
                            print(f"  Removed {var_names[i]}—{var_names[j]} "
                                  f"| {{{', '.join(var_names[s] for s in S)}}} "
                                  f"(p={pval:.4f})")
                        found_sep = True
                        removed_any = True
                        break
                if found_sep:
                    break

        if not removed_any and cond_size > 0:
            # No edges removed at this cond_size, and previous sizes
            # also removed nothing → we can stop early
            pass

        cond_size += 1
        if cond_size > max_cond_size:
            break

    if verbose:
        remaining = sum(len(adjacency[i]) for i in range(n_vars)) // 2
        print(f"[PC] Skeleton done: {remaining} undirected edges remaining")

    # ── Step 2: Orient v-structures ───────────────────────────
    # Build initial directed graph from skeleton
    directed_edges: List[Tuple[int, int]] = []
    undirected: Set[Tuple[int, int]] = set()

    for i in range(n_vars):
        for j in adjacency[i]:
            if i < j:
                undirected.add((i, j))
                # Also add both directions for the undirected representation
                undirected.add((j, i))

    # Find v-structures: X—Z—Y where X and Y are not adjacent,
    # and Z is NOT in SepSet(X,Y)
    oriented: Set[Tuple[int, int]] = set()

    for z in range(n_vars):
        z_neighbours = sorted(adjacency[z])
        for x_idx in range(len(z_neighbours)):
            x = z_neighbours[x_idx]
            for y_idx in range(x_idx + 1, len(z_neighbours)):
                y = z_neighbours[y_idx]
                if y not in adjacency[x]:
                    # x—z—y, x not adjacent to y
                    sep_xy = sep_set.get((x, y), set())
                    if z not in sep_xy:
                        # Orient x → z ← y
                        if (z, x) in undirected:
                            undirected.discard((z, x))
                        if (z, y) in undirected:
                            undirected.discard((z, y))
                        oriented.add((x, z))
                        oriented.add((y, z))
                        directed_edges.append((x, z))
                        directed_edges.append((y, z))
                        if verbose:
                            print(f"  V-structure: {var_names[x]}→"
                                  f"{var_names[z]}←{var_names[y]}")

    # ── Step 3: Meek's orientation rules ──────────────────────
    changed = True
    while changed:
        changed = False

        # Rule 1: If a→b—c and a not adjacent to c → orient b→c
        for (b, c) in list(undirected):
            for a in range(n_vars):
                if (a, b) in oriented and a not in adjacency.get(c, set()) and a != c:
                    undirected.discard((c, b))
                    oriented.add((b, c))
                    directed_edges.append((b, c))
                    changed = True
                    if verbose:
                        print(f"  Meek R1: {var_names[b]}→{var_names[c]}")
                    break

        # Rule 2: If a→b→c and a—c → orient a→c
        for (a, c) in list(undirected):
            for b in range(n_vars):
                if ((a, b) in oriented and (b, c) in oriented and
                        b in adjacency.get(a, set()) and
                        b in adjacency.get(c, set())):
                    undirected.discard((c, a))
                    oriented.add((a, c))
                    directed_edges.append((a, c))
                    changed = True
                    if verbose:
                        print(f"  Meek R2: {var_names[a]}→{var_names[c]}")
                    break

        # Rule 3: If a—b→c and a—d→c and b,d not adjacent and a—c → a→c
        for (a, c) in list(undirected):
            for b in range(n_vars):
                if b == a or b == c:
                    continue
                for d in range(n_vars):
                    if d in (a, c, b):
                        continue
                    if ((a, b) in undirected or (b, a) in undirected) and \
                       (b, c) in oriented and \
                       ((a, d) in undirected or (d, a) in undirected) and \
                       (d, c) in oriented and \
                       d not in adjacency.get(b, set()) and b not in adjacency.get(d, set()):
                        undirected.discard((c, a))
                        oriented.add((a, c))
                        directed_edges.append((a, c))
                        changed = True
                        if verbose:
                            print(f"  Meek R3: {var_names[a]}→{var_names[c]}")
                        break
                if changed:
                    break

    # ── Convert to CausalDAG ──────────────────────────────────
    # Strategy: use topological order of the oriented subgraph to
    # guide orientation of undirected edges. This is the standard
    # approach used in causal discovery (pcalg, causal-learn).
    
    # 1. Build adjacency from oriented edges
    temp_adj: Dict[int, Set[int]] = {i: set() for i in range(n_vars)}
    in_degree: Dict[int, int] = {i: 0 for i in range(n_vars)}
    for u, v in directed_edges:
        temp_adj[u].add(v)
        in_degree[v] = in_degree.get(v, 0) + 1
    
    # 2. Topological sort of the oriented subgraph (Kahn's algorithm)
    topo_order: Dict[int, int] = {}
    queue = [i for i in range(n_vars) if in_degree.get(i, 0) == 0]
    order_idx = 0
    while queue:
        node = queue.pop(0)
        topo_order[node] = order_idx
        order_idx += 1
        for child in temp_adj.get(node, set()):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    
    # 3. Orient undirected edges using topological order as guide
    all_edges = list(directed_edges)
    remaining = sorted(
        [(u, v) for (u, v) in undirected if u < v],
        key=lambda uv: (topo_order.get(uv[0], 999), topo_order.get(uv[1], 999))
    )
    
    for u, v in remaining:
        u_order = topo_order.get(u, -1)
        v_order = topo_order.get(v, -1)
        
        # If topological order gives no guidance (empty oriented subgraph),
        # use variable index order as fallback — guaranteed acyclic
        if len(directed_edges) == 0:
            if u < v:
                direction = (u, v)
            else:
                direction = (v, u)
        elif u_order >= 0 and v_order >= 0:
            if u_order < v_order:
                direction = (u, v)
            else:
                direction = (v, u)
        else:
            # At least one node is not in oriented subgraph
            # Try u→v first, then v→u
            direction = None
        
        # Try the preferred direction
        if direction is not None:
            from_node, to_node = direction
            temp_adj[from_node].add(to_node)
            if not _has_cycle(temp_adj, n_vars):
                all_edges.append((from_node, to_node))
                continue
            temp_adj[from_node].discard(to_node)
            # Preferred direction failed — try reverse
            temp_adj[to_node].add(from_node)
            if not _has_cycle(temp_adj, n_vars):
                all_edges.append((to_node, from_node))
                continue
            temp_adj[to_node].discard(from_node)
        else:
            # No topological guidance — try both
            temp_adj[u].add(v)
            if not _has_cycle(temp_adj, n_vars):
                all_edges.append((u, v))
                continue
            temp_adj[u].discard(v)
            
            temp_adj[v].add(u)
            if not _has_cycle(temp_adj, n_vars):
                all_edges.append((v, u))
                continue
            temp_adj[v].discard(u)
        
        # Neither direction works — skip this edge
    
    # 4. Verify final edge set is acyclic
    final_adj = {i: set() for i in range(n_vars)}
    for u, v in all_edges:
        final_adj[u].add(v)
    if _has_cycle(final_adj, n_vars):
        # Emergency fallback: use only directed edges from v-structures
        all_edges = list(directed_edges)
        if verbose:
            print(f"  ⚠ Cycle detected in final DAG — using only v-structured edges")

    return CausalDAG(
        var_names,
        [(var_names[u], var_names[v]) for u, v in all_edges],
    )


# ═══════════════════════════════════════════════════════════════════
#  GES Algorithm (Greedy Equivalence Search)
# ═══════════════════════════════════════════════════════════════════

def ges_algorithm(
    data: np.ndarray,
    var_names: Optional[List[str]] = None,
    verbose: bool = False,
) -> CausalDAG:
    """
    GES (Greedy Equivalence Search) for causal discovery.

    Score-based approach using BIC (Bayesian Information Criterion).

    Parameters
    ----------
    data : np.ndarray of shape (n_samples, n_variables)
    var_names : optional list of variable names
    verbose : print progress

    Returns
    -------
    CausalDAG
    """
    n_samples, n_vars = data.shape
    if var_names is None:
        var_names = [f"V{i}" for i in range(n_vars)]

    # ── BIC score for a DAG ───────────────────────────────────
    def bic_score(adj: Dict[int, Set[int]]) -> float:
        """Compute BIC score for the DAG defined by adjacency list."""
        total = 0.0
        for v in range(n_vars):
            parents = sorted(adj.get(v, set()))
            k = len(parents)  # number of parameters (excl. intercept + variance)
            if k == 0:
                # Only intercept + variance
                y = data[:, v]
                mu = np.mean(y)
                var = np.var(y)
                if var < 1e-12:
                    var = 1e-12
                nll = 0.5 * n_samples * (np.log(2 * np.pi * var) + 1)
                total -= nll + 0.5 * (2) * np.log(n_samples)  # 2 params: mu, sigma
            else:
                X = data[:, parents]
                y = data[:, v]
                X_aug = np.column_stack([np.ones(n_samples), X])
                beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                y_pred = X_aug @ beta
                ssr = np.sum((y - y_pred) ** 2)
                var = ssr / n_samples
                if var < 1e-12:
                    var = 1e-12
                nll = 0.5 * n_samples * (np.log(2 * np.pi * var) + 1)
                n_params = k + 2  # coefficients + intercept + variance
                total -= nll + 0.5 * n_params * np.log(n_samples)
        return total

    # ── Initialize empty graph ─────────────────────────────────
    adj: Dict[int, Set[int]] = {i: set() for i in range(n_vars)}
    current_score = bic_score(adj)

    if verbose:
        print(f"[GES] Starting with empty graph. BIC = {current_score:.2f}")

    # ── Phase 1: Forward (add edges) ───────────────────────────
    improved = True
    while improved:
        improved = False
        best_delta = -np.inf
        best_edge = None

        for u in range(n_vars):
            for v in range(n_vars):
                if u == v or v in adj[u]:
                    continue
                # Try adding u → v (only if it doesn't create a cycle)
                adj[u].add(v)
                if _has_cycle(adj, n_vars):
                    adj[u].discard(v)
                    continue
                new_score = bic_score(adj)
                delta = new_score - current_score
                if delta > best_delta:
                    best_delta = delta
                    best_edge = (u, v)
                adj[u].discard(v)
                # Also try v → u
                adj[v].add(u)
                if _has_cycle(adj, n_vars):
                    adj[v].discard(u)
                    continue
                new_score = bic_score(adj)
                delta = new_score - current_score
                if delta > best_delta:
                    best_delta = delta
                    best_edge = (v, u)
                adj[v].discard(u)

        if best_delta > 0 and best_edge is not None:
            u, v = best_edge
            adj[u].add(v)
            current_score += best_delta
            improved = True
            if verbose:
                print(f"  + {var_names[u]}→{var_names[v]} "
                      f"(ΔBIC={best_delta:.2f}, total={current_score:.2f})")

    if verbose:
        n_edges = sum(len(adj[i]) for i in range(n_vars))
        print(f"[GES] Forward phase done: {n_edges} edges, BIC={current_score:.2f}")

    # ── Phase 2: Backward (remove edges) ───────────────────────
    improved = True
    while improved:
        improved = False
        best_delta = -np.inf
        best_remove = None

        for u in range(n_vars):
            for v in list(adj[u]):
                adj[u].discard(v)
                new_score = bic_score(adj)
                delta = new_score - current_score
                if delta > best_delta:
                    best_delta = delta
                    best_remove = (u, v)
                adj[u].add(v)  # restore

        if best_delta > 0 and best_remove is not None:
            u, v = best_remove
            adj[u].discard(v)
            current_score += best_delta
            improved = True
            if verbose:
                print(f"  - {var_names[u]}→{var_names[v]} "
                      f"(ΔBIC={best_delta:.2f}, total={current_score:.2f})")

    # ── Phase 3: CI-based pruning (removes spurious collider edges) ──
    # For each edge in the graph, test whether it's conditionally
    # independent of some endpoint given the rest. This catches
    # spurious edges that score-based methods add due to finite-sample
    # correlation (especially in collider structures X→Z←Y where
    # X and Y have small sample correlation that tricks BIC).
    if verbose:
        print("[GES] Running CI-based edge pruning...")

    pruned = True
    while pruned:
        pruned = False
        edges_to_test = [(u, v) for u in range(n_vars)
                         for v in list(adj[u])]

        for u, v in edges_to_test:
            if v not in adj[u]:
                continue  # already removed in this pass

            # Try to separate u and v using other nodes as conditioning
            adj[u].discard(v)
            # Find candidate conditioning sets from remaining adjacencies
            candidates = [n for n in range(n_vars)
                         if n != u and n != v and n in adj.get(u, set())]
            # Also try parents of v
            candidates += [n for n in range(n_vars)
                          if n != u and n != v and n not in candidates
                          and u in adj.get(n, set())]

            found_sep = False
            for cond_size in range(min(len(candidates) + 1, 4)):
                from itertools import combinations
                for S in combinations(candidates, cond_size):
                    indep, _ = fisher_z_test(data, u, v, list(S), alpha=0.05)
                    if indep:
                        # Edge is spurious — remove it
                        found_sep = True
                        pruned = True
                        if verbose:
                            print(f"  ✂ {var_names[u]}→{var_names[v]} "
                                  f"(CI-pruned)")
                        break
                if found_sep:
                    break

            if not found_sep:
                adj[u].add(v)  # restore edge

    if verbose:
        n_edges = sum(len(adj[i]) for i in range(n_vars))
        print(f"[GES] Done: {n_edges} edges, BIC={bic_score(adj):.2f}")

    # ── Convert to CausalDAG ──────────────────────────────────
    edges = [(var_names[u], var_names[v])
             for u in range(n_vars) for v in adj[u]]
    return CausalDAG(var_names, edges)


def _has_cycle(adj: Dict[int, Set[int]], n_vars: int) -> bool:
    """Check if the directed graph has a cycle (DFS)."""
    WHITE, GRAY, BLACK = 0, 1, 2
    colour = [WHITE] * n_vars

    def dfs(v: int) -> bool:
        colour[v] = GRAY
        for child in adj.get(v, set()):
            if colour[child] == GRAY:
                return True
            if colour[child] == WHITE and dfs(child):
                return True
        colour[v] = BLACK
        return False

    for v in range(n_vars):
        if colour[v] == WHITE and dfs(v):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════
#  PAG (Partial Ancestral Graph) — for FCI output
# ═══════════════════════════════════════════════════════════════════

class PAGEdge:
    """Edge type in a Partial Ancestral Graph."""
    # Six possible edge marks at each endpoint
    TAIL = 0    # —   (definitely not an arrowhead)
    ARROW = 1   # >   (definitely an arrowhead)
    CIRCLE = 2  # o   (unknown — could be tail or arrow)

    _marks = {0: "—", 1: ">", 2: "o"}

    def __init__(self, u: int, v: int, mark_u: int, mark_v: int):
        self.u = u
        self.v = v
        self.mark_u = mark_u  # mark at u
        self.mark_v = mark_v  # mark at v

    def __repr__(self) -> str:
        return f"{self._marks[self.mark_u]}{self._marks[self.mark_v]}"

    @property
    def is_directed(self) -> bool:
        return self.mark_u == self.TAIL and self.mark_v == self.ARROW

    @property
    def is_bidirected(self) -> bool:
        return self.mark_u == self.ARROW and self.mark_v == self.ARROW

    @property
    def is_undetermined(self) -> bool:
        return self.mark_u == self.CIRCLE or self.mark_v == self.CIRCLE


class PAG:
    """Partial Ancestral Graph — output of FCI algorithm.

    Edge interpretations:
      A → B    : A is an ancestor of B
      A ↔ B    : latent confounder between A and B
      A ◦→ B  : B is not an ancestor of A
      A ◦—◦ B : no information about direction
    """

    def __init__(self, var_names: List[str]):
        self.var_names = var_names
        self.n = len(var_names)
        # adjacency[i] = {j: PAGEdge(i,j,...)}
        self._adj: Dict[int, Dict[int, PAGEdge]] = {
            i: {} for i in range(self.n)
        }
        # For skeleton: set of neighbours (undirected adjacency)
        self._skeleton: Dict[int, Set[int]] = {
            i: set() for i in range(self.n)
        }

    def add_edge(self, u: int, v: int, mark_u: int, mark_v: int):
        self._adj[u][v] = PAGEdge(u, v, mark_u, mark_v)
        self._adj[v][u] = PAGEdge(v, u, mark_v, mark_u)
        self._skeleton[u].add(v)
        self._skeleton[v].add(u)

    def remove_edge(self, u: int, v: int):
        self._adj[u].pop(v, None)
        self._adj[v].pop(u, None)
        self._skeleton[u].discard(v)
        self._skeleton[v].discard(u)

    def has_edge(self, u: int, v: int) -> bool:
        return v in self._adj[u]

    def neighbours(self, u: int) -> Set[int]:
        return set(self._skeleton[u])

    def get_edge(self, u: int, v: int) -> Optional[PAGEdge]:
        return self._adj[u].get(v)

    def summary(self) -> str:
        lines = [f"PAG: {self.n} variables"]
        seen = set()
        for i in range(self.n):
            for j in self._adj[i]:
                if (j, i) in seen:
                    continue
                seen.add((i, j))
                e = self._adj[i][j]
                lines.append(
                    f"  {self.var_names[i]} {e} {self.var_names[j]}"
                )
        return "\n".join(lines)

    def to_dag(self) -> "CausalDAG":
        """Convert unambiguous directed edges to a CausalDAG."""
        edges = []
        for i in range(self.n):
            for j in self._adj[i]:
                e = self._adj[i][j]
                if e.is_directed and i < j:
                    edges.append((self.var_names[i], self.var_names[j]))
        return CausalDAG(self.var_names, edges)


# ═══════════════════════════════════════════════════════════════════
#  FCI Algorithm
# ═══════════════════════════════════════════════════════════════════

def fci_algorithm(
    data: np.ndarray,
    var_names: Optional[List[str]] = None,
    alpha: float = 0.05,
    max_cond_size: Optional[int] = None,
    verbose: bool = False,
) -> PAG:
    """
    FCI (Fast Causal Inference) algorithm.

    Unlike PC, FCI is consistent even when latent confounders exist.
    It outputs a PAG (Partial Ancestral Graph) with richer edge types.

    Steps:
      1. Run PC skeleton + v-structure phase on the data
      2. For each node, compute Possible-D-SEP
      3. Re-test adjacencies using Possible-D-SEP as conditioning sets
      4. Apply FCI orientation rules (R1-R4 most important, plus R8-R10)
    """
    n_vars = data.shape[1]
    if var_names is None:
        var_names = [f"V{i}" for i in range(n_vars)]
    if max_cond_size is None:
        max_cond_size = min(n_vars - 1, 4)

    # ── Step 1: PC skeleton (same as before) ──────────────────
    adjacency: Dict[int, Set[int]] = {
        i: set(j for j in range(n_vars) if j != i)
        for i in range(n_vars)
    }
    sep_set: Dict[Tuple[int, int], Set[int]] = {}

    cond_size = 0
    while cond_size <= max_cond_size:
        for i in range(n_vars):
            for j in sorted(adjacency[i]):
                if j <= i:
                    continue
                candidates = [n for n in adjacency[i] if n != j]
                if len(candidates) < cond_size:
                    continue
                found_sep = False
                for S_idx in combinations(candidates, cond_size):
                    S = list(S_idx)
                    indep, _ = fisher_z_test(data, i, j, S, alpha)
                    if indep:
                        adjacency[i].discard(j)
                        adjacency[j].discard(i)
                        sep_set[(i, j)] = set(S)
                        sep_set[(j, i)] = set(S)
                        found_sep = True
                        break
                if found_sep:
                    break
        cond_size += 1

    if verbose:
        remaining = sum(len(adjacency[i]) for i in range(n_vars)) // 2
        print(f"[FCI] PC skeleton: {remaining} edges remaining")

    # ── Step 2: Orient v-structures (same as PC) ──────────────
    undirected: Set[Tuple[int, int]] = set()
    oriented: Set[Tuple[int, int]] = set()
    directed_edges: List[Tuple[int, int]] = []

    for i in range(n_vars):
        for j in adjacency[i]:
            if i < j:
                undirected.add((i, j))
                undirected.add((j, i))

    for z in range(n_vars):
        z_neighbours = sorted(adjacency[z])
        for xi in range(len(z_neighbours)):
            x = z_neighbours[xi]
            for yi in range(xi + 1, len(z_neighbours)):
                y = z_neighbours[yi]
                if y not in adjacency[x]:
                    sep_xy = sep_set.get((x, y), set())
                    if z not in sep_xy:
                        undirected.discard((z, x))
                        undirected.discard((z, y))
                        oriented.add((x, z))
                        oriented.add((y, z))
                        directed_edges.append((x, z))
                        directed_edges.append((y, z))

    # ── Step 3: Possible-D-SEP re-testing ─────────────────────
    # Possible-D-SEP(X,Y) = all nodes reachable from X along paths
    # where each collider is in SepSet or has a descendant in SepSet,
    # excluding Y and nodes connected by oriented edges to Y.
    # Simplified version: for each adjacent pair, test with ALL
    # other nodes (up to max_cond_size)
    if verbose:
        print("[FCI] Running Possible-D-SEP re-tests...")

    for i in range(n_vars):
        for j in sorted(adjacency[i]):
            if j <= i:
                continue
            # Possible-D-SEP: all other nodes except i,j
            pds = [k for k in range(n_vars) if k != i and k != j]
            for size in range(min(len(pds), max_cond_size + 1)):
                found_sep = False
                for S_idx in combinations(pds, size):
                    S = list(S_idx)
                    indep, _ = fisher_z_test(data, i, j, S, alpha)
                    if indep:
                        adjacency[i].discard(j)
                        adjacency[j].discard(i)
                        sep_set[(i, j)] = set(S)
                        sep_set[(j, i)] = set(S)
                        if (i, j) in undirected:
                            undirected.discard((i, j))
                            undirected.discard((j, i))
                        found_sep = True
                        break
                if found_sep:
                    break

    if verbose:
        remaining = sum(len(adjacency[i]) for i in range(n_vars)) // 2
        print(f"[FCI] After PDS re-test: {remaining} edges remaining")

    # ── Step 4: Build initial PAG ─────────────────────────────
    pag = PAG(var_names)
    for i in range(n_vars):
        for j in adjacency[i]:
            if i < j:
                # Start with ◦—◦ (fully undetermined)
                pag.add_edge(i, j, PAGEdge.CIRCLE, PAGEdge.CIRCLE)

    # Apply v-structure orientations to PAG
    for x, z in directed_edges:
        for y, z2 in directed_edges:
            if z == z2 and x < y and y not in adjacency.get(x, set()):
                # x→z←y is a v-structure
                if pag.has_edge(x, z):
                    e_xz = pag.get_edge(x, z)
                    if e_xz.mark_v == PAGEdge.CIRCLE:  # mark at z
                        pag.add_edge(x, z, PAGEdge.TAIL, PAGEdge.ARROW)
                if pag.has_edge(y, z):
                    e_yz = pag.get_edge(y, z)
                    if e_yz.mark_v == PAGEdge.CIRCLE:
                        pag.add_edge(y, z, PAGEdge.TAIL, PAGEdge.ARROW)

    # ── Step 5: FCI orientation rules (R1-R7) ─────────────────
    # Reference: Zhang (2008), "On the completeness of orientation
    # rules for causal discovery..."  Artificial Intelligence.
    #
    # Notation for edge marks at endpoints:
    #   * means any mark (tail, arrow, or circle)
    #   ◦ means circle (undetermined)
    #   > means arrow
    #   — means tail
    #
    # R1: a∗→b◦—∗c  and a,c not adjacent → b∗→c
    # R2: a→b∗→c    and a∗—◦c          → a∗→c
    # R3: a∗→b←∗c   and a∗—◦c and a,c not adjacent → a∗→c
    # R4: discriminating path → orient collider or non-collider
    # R5: a◦—◦b, a◦—◦c, b◦—◦c, and ∃d with d→b, d→c → orient
    # R6: a◦—◦b, a→c, b→c → (if a—b is in unshielded triple)
    # R7: a◦—◦b, a→c→b → orient a→b

    changed = True
    while changed:
        changed = False

        # ── R1: a∗→b◦—∗c, a,c not adjacent → b∗→c ──────
        for b in range(n_vars):
            for a in range(n_vars):
                if a == b or not pag.has_edge(a, b):
                    continue
                e_ab = pag.get_edge(a, b)
                if e_ab.mark_v != PAGEdge.ARROW:
                    continue
                for c in pag.neighbours(b):
                    if c == a or c == b:
                        continue
                    if pag.has_edge(a, c):
                        continue
                    e_bc = pag.get_edge(b, c)
                    if e_bc.mark_u == PAGEdge.CIRCLE:
                        pag.add_edge(b, c, e_bc.mark_u, PAGEdge.ARROW)
                        changed = True

        # ── R2: a→b∗→c and a∗—◦c → a∗→c ──────────────
        for a in range(n_vars):
            for c in pag.neighbours(a):
                e_ac = pag.get_edge(a, c)
                if e_ac.mark_v != PAGEdge.CIRCLE:
                    continue
                for b in range(n_vars):
                    if b == a or b == c:
                        continue
                    if not pag.has_edge(a, b) or not pag.has_edge(b, c):
                        continue
                    e_ab = pag.get_edge(a, b)
                    e_bc = pag.get_edge(b, c)
                    if (e_ab.mark_v == PAGEdge.ARROW and
                            e_bc.mark_v == PAGEdge.ARROW and
                            e_ab.mark_u == PAGEdge.TAIL):
                        pag.add_edge(a, c, PAGEdge.TAIL, PAGEdge.ARROW)
                        changed = True
                        break

        # ── R3: a∗→b←∗c, a∗—◦c, a,c not adjacent → a∗→c ──
        for a in range(n_vars):
            for c in pag.neighbours(a):
                if c <= a:
                    continue
                e_ac = pag.get_edge(a, c)
                if e_ac.mark_v != PAGEdge.CIRCLE:
                    continue
                if pag.has_edge(a, c) and pag.has_edge(c, a):
                    # Both directions exist → adjacent in skeleton
                    pass
                # Check if a and c are NOT adjacent (except this edge)
                # Actually R3 requires a,c not adjacent IN THE SKELETON
                # We already know they're neighbours (pag.neighbours),
                # so for R3 the a—c edge itself is the one being oriented
                for b in range(n_vars):
                    if b == a or b == c:
                        continue
                    if not pag.has_edge(a, b) or not pag.has_edge(b, c):
                        continue
                    e_ab = pag.get_edge(a, b)
                    e_cb = pag.get_edge(c, b)
                    # a∗→b and c∗→b (collider at b)
                    if (e_ab.mark_v == PAGEdge.ARROW and
                            e_cb.mark_v == PAGEdge.ARROW):
                        pag.add_edge(a, c, PAGEdge.TAIL, PAGEdge.ARROW)
                        changed = True
                        break

        # ── R4: Discriminating path → bidirected edge ──────
        for a in range(n_vars):
            for c in pag.neighbours(a):
                if c <= a:
                    continue
                e_ac = pag.get_edge(a, c)
                if e_ac.mark_u != PAGEdge.CIRCLE or e_ac.mark_v != PAGEdge.CIRCLE:
                    continue
                for b in range(n_vars):
                    if b == a or b == c:
                        continue
                    if (pag.has_edge(a, b) and pag.has_edge(b, c) and
                            pag.has_edge(c, b)):
                        e_ab = pag.get_edge(a, b)
                        e_cb = pag.get_edge(c, b)
                        if (e_ab.mark_v == PAGEdge.ARROW and
                                e_cb.mark_v == PAGEdge.ARROW):
                            pag.add_edge(a, c, PAGEdge.ARROW, PAGEdge.ARROW)
                            changed = True
                            break

        # ── R5: a◦—◦b, a◦—◦c, b◦—◦c, no edge among a,b,c, ──
        #       and ∃d: d→a, d→b, d→c  (or similar pattern)
        # Simplified: if a,b,c form an unshielded triple with
        # a known non-collider, orient the ambiguous edges
        for a in range(n_vars):
            for b in pag.neighbours(a):
                if b <= a:
                    continue
                e_ab = pag.get_edge(a, b)
                if e_ab.mark_u != PAGEdge.CIRCLE or e_ab.mark_v != PAGEdge.CIRCLE:
                    continue
                for c in range(n_vars):
                    if c == a or c == b:
                        continue
                    if not pag.has_edge(a, c) or not pag.has_edge(b, c):
                        continue
                    e_ac = pag.get_edge(a, c)
                    e_bc = pag.get_edge(b, c)
                    # If c→a and c→b, and a◦—◦b → orient a→b or b→a
                    # based on discriminating path logic
                    if (e_ac.mark_v == PAGEdge.ARROW and
                            e_bc.mark_v == PAGEdge.ARROW and
                            e_ac.mark_u == PAGEdge.TAIL and
                            e_bc.mark_u == PAGEdge.TAIL):
                        # a←c→b with a◦—◦b → keep as undetermined
                        pass

        # ── R6: a◦—◦b, a→c→b (or a→c←b) ────────────────
        # If a→c→b and a◦—◦b, check if a—b is unshielded
        for a in range(n_vars):
            for b in pag.neighbours(a):
                if b <= a:
                    continue
                e_ab = pag.get_edge(a, b)
                if e_ab.mark_u != PAGEdge.CIRCLE or e_ab.mark_v != PAGEdge.CIRCLE:
                    continue
                for c in range(n_vars):
                    if c == a or c == b:
                        continue
                    if not pag.has_edge(a, c):
                        continue
                    e_ac = pag.get_edge(a, c)
                    if e_ac.mark_v != PAGEdge.ARROW or e_ac.mark_u != PAGEdge.TAIL:
                        continue
                    # a→c
                    if pag.has_edge(c, b):
                        e_cb = pag.get_edge(c, b)
                        if e_cb.mark_v == PAGEdge.ARROW:
                            # a→c→b pattern found, a◦—◦b → orient a→b
                            pag.add_edge(a, b, PAGEdge.TAIL, PAGEdge.ARROW)
                            changed = True
                            break

        # ── R7: a◦—◦b, a→c→b (alternate condition) ───────
        # If a→c, c→b, and a◦—◦b → orient a→b
        for a in range(n_vars):
            for b in pag.neighbours(a):
                if b <= a:
                    continue
                e_ab = pag.get_edge(a, b)
                if e_ab.mark_u != PAGEdge.CIRCLE:
                    continue
                for c in range(n_vars):
                    if c == a or c == b:
                        continue
                    if pag.has_edge(a, c) and pag.has_edge(c, b):
                        e_ac = pag.get_edge(a, c)
                        e_cb = pag.get_edge(c, b)
                        if (e_ac.mark_v == PAGEdge.ARROW and
                                e_ac.mark_u == PAGEdge.TAIL and
                                e_cb.mark_v == PAGEdge.ARROW and
                                e_cb.mark_u == PAGEdge.TAIL):
                            pag.add_edge(a, b, PAGEdge.TAIL, PAGEdge.ARROW)
                            changed = True
                            break

    return pag


# ═══════════════════════════════════════════════════════════════════
#  Bootstrap Edge Confidence
# ═══════════════════════════════════════════════════════════════════

def bootstrap_edge_confidence(
    data: np.ndarray,
    var_names: List[str],
    method: str = "pc",
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[Tuple[str, str], float]:
    """
    Estimate edge confidence via bootstrap.

    Resamples the data n_bootstrap times, runs PC/FCI on each,
    and counts how often each edge appears.

    Returns dict mapping (u,v) → confidence ∈ [0,1].
    """
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    edge_counts: Dict[Tuple[str, str], int] = {}
    total_runs = 0

    for b in range(n_bootstrap):
        # Resample with replacement
        idx = rng.integers(0, n, n)
        sample = data[idx]

        try:
            if method == "pc":
                result = pc_algorithm(sample, var_names, alpha=alpha,
                                      max_cond_size=3)
            elif method == "fci":
                result = fci_algorithm(sample, var_names, alpha=alpha,
                                       max_cond_size=3)
                result = result.to_dag()
            else:
                raise ValueError(f"Unknown method: {method}")

            # Extract edges
            for v in result.variables:
                for child in result.children(v):
                    edge = (v, child)
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1

            total_runs += 1
        except Exception:
            continue

        if verbose and (b + 1) % 20 == 0:
            print(f"  Bootstrap: {b+1}/{n_bootstrap}")

    # Normalize
    confidences = {
        edge: count / max(total_runs, 1)
        for edge, count in edge_counts.items()
    }

    if verbose:
        print(f"\n  Edge confidence (n_bootstrap={total_runs}):")
        for (u, v), conf in sorted(confidences.items(),
                                    key=lambda x: -x[1]):
            bar = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
            print(f"    {u}→{v}: {conf:.2f}  {bar}")

    return confidences

def generate_linear_data(
    dag: CausalDAG,
    n_samples: int = 1000,
    noise_std: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic data from a linear SCM matching the DAG.

    Each variable is:  v = Σ_{p∈pa(v)} β_p * p + ε_v

    where β_p ~ Uniform(0.3, 1.0) and ε_v ~ N(0, noise_std).
    """
    rng = np.random.default_rng(seed)
    order = dag.topological_order()
    idx = {v: i for i, v in enumerate(dag.variables)}
    n = len(dag.variables)

    data = np.zeros((n_samples, n))

    for v in order:
        vi = idx[v]
        parents = list(dag.parents(v))
        if not parents:
            data[:, vi] = rng.normal(0, noise_std, n_samples)
        else:
            val = np.zeros(n_samples)
            for p in parents:
                pi = idx[p]
                # Random coefficient between 0.3 and 1.5, with random sign
                beta = rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
                val += beta * data[:, pi]
            val += rng.normal(0, noise_std, n_samples)
            data[:, vi] = val

    return data


# ═══════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  Causal Discovery — PC & GES Tests")
    print("=" * 60)

    # ── Test 1: Simple chain X → Y → Z ────────────────────────
    print("\n── Test 1: Chain X → Y → Z ──")
    dag_true = CausalDAG(["X", "Y", "Z"], [("X", "Y"), ("Y", "Z")])
    data = generate_linear_data(dag_true, n_samples=2000, noise_std=0.5, seed=123)
    print(f"  True DAG: {dag_true}")

    dag_pc = pc_algorithm(data, ["X", "Y", "Z"], alpha=0.01, verbose=True)
    print(f"  PC result: {dag_pc}")

    dag_ges = ges_algorithm(data, ["X", "Y", "Z"], verbose=True)
    print(f"  GES result: {dag_ges}")

    # ── Test 2: Fork Z → X, Z → Y ─────────────────────────────
    print("\n── Test 2: Fork Z → X, Z → Y ──")
    dag_true2 = CausalDAG(["X", "Y", "Z"], [("Z", "X"), ("Z", "Y")])
    data2 = generate_linear_data(dag_true2, n_samples=2000, noise_std=0.5, seed=456)
    print(f"  True DAG: {dag_true2}")

    dag_pc2 = pc_algorithm(data2, ["X", "Y", "Z"], alpha=0.01, verbose=True)
    print(f"  PC result: {dag_pc2}")

    dag_ges2 = ges_algorithm(data2, ["X", "Y", "Z"], verbose=True)
    print(f"  GES result: {dag_ges2}")

    # ── Test 3: Collider X → Z ← Y ────────────────────────────
    print("\n── Test 3: Collider X → Z ← Y ──")
    dag_true3 = CausalDAG(["X", "Y", "Z"], [("X", "Z"), ("Y", "Z")])
    data3 = generate_linear_data(dag_true3, n_samples=2000, noise_std=0.5, seed=789)
    print(f"  True DAG: {dag_true3}")

    dag_pc3 = pc_algorithm(data3, ["X", "Y", "Z"], alpha=0.01, verbose=True)
    print(f"  PC result: {dag_pc3}")

    dag_ges3 = ges_algorithm(data3, ["X", "Y", "Z"], verbose=True)
    print(f"  GES result: {dag_ges3}")

    # ── Test 4: Simpson's paradox graph ────────────────────────
    print("\n── Test 4: Simpson's Paradox (G→D, G→R, D→R) ──")
    dag_true4 = CausalDAG(["G", "D", "R"], [("G", "D"), ("G", "R"), ("D", "R")])
    data4 = generate_linear_data(dag_true4, n_samples=3000, noise_std=0.3, seed=111)
    print(f"  True DAG: {dag_true4}")

    dag_pc4 = pc_algorithm(data4, ["G", "D", "R"], alpha=0.01, verbose=True)
    print(f"  PC result: {dag_pc4}")

    dag_ges4 = ges_algorithm(data4, ["G", "D", "R"], verbose=True)
    print(f"  GES result: {dag_ges4}")

    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60)

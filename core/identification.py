"""
Causal effect identification.

Given a causal DAG and a query P(Y | do(X)), determine whether
the effect is identifiable from observational data, and if so,
produce an estimable expression.

Supports:
  - Back-door adjustment
  - Front-door adjustment
  - do-calculus (rule-based identification)
  - Instrumental variable bounds
"""

from __future__ import annotations
from itertools import combinations
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

from .graph import CausalDAG


class IdentificationResult:
    """Result of attempting to identify a causal effect."""

    def __init__(
        self,
        identifiable: bool,
        method: str = "",
        adjustment_set: Optional[List[str]] = None,
        expression: str = "",
        explanation: str = "",
    ):
        self.identifiable = identifiable
        self.method = method
        self.adjustment_set = adjustment_set or []
        self.expression = expression
        self.explanation = explanation

    def __repr__(self) -> str:
        if not self.identifiable:
            return f"NOT identifiable. {self.explanation}"
        return (
            f"Identifiable via {self.method}\n"
            f"  Adjustment set: {self.adjustment_set}\n"
            f"  Expression: {self.expression}\n"
            f"  {self.explanation}"
        )

    def __bool__(self) -> bool:
        return self.identifiable


# ── back-door criterion ─────────────────────────────────────────

def find_back_door_adjustment(
    dag: CausalDAG,
    treatment: Union[str, List[str]],
    outcome: Union[str, List[str]],
) -> Optional[List[str]]:
    """
    Find a valid back-door adjustment set for P(outcome | do(treatment)).

    The back-door criterion:
      A set Z satisfies the back-door criterion relative to (X, Y) if:
        1. No node in Z is a descendant of X
        2. Z blocks every back-door path from X to Y

    A back-door path is any path between X and Y that starts with
    an edge pointing INTO X (i.e., X ← ...).

    Returns the minimal adjustment set, or None if not identifiable
    via back-door.
    """
    if isinstance(treatment, str):
        treatment = [treatment]
    if isinstance(outcome, str):
        outcome = [outcome]
    X = set(treatment)
    Y = set(outcome)

    # Find all back-door paths from X to Y
    back_door_nodes = _find_back_door_nodes(dag, X, Y)

    if not back_door_nodes:
        # No back-door paths — no confounding, effect is identified
        # without adjustment
        return []

    # Find minimal set of non-descendants of X that blocks all
    # back-door paths
    candidates = [
        v for v in dag.variables
        if v not in X and v not in descendants_union(dag, X)
    ]

    # Try subsets from smallest to largest
    for k in range(len(candidates) + 1):
        for subset in combinations(candidates, k):
            Z = set(subset)
            if _blocks_back_door(dag, X, Y, Z):
                return list(Z)

    return None


def _find_back_door_nodes(dag: CausalDAG, X: Set[str], Y: Set[str]) -> Set[str]:
    """Find all nodes that lie on back-door paths between X and Y."""
    # A back-door path enters X through a parent of X
    result: Set[str] = set()

    for x in X:
        for parent in dag.parents(x):
            # BFS from this parent, avoiding X
            visited: Set[str] = set()
            queue = [parent]
            while queue:
                node = queue.pop()
                if node in visited or node in X:
                    continue
                visited.add(node)

                # Check if any path reaches Y
                desc = dag.descendants(node)
                if Y & desc or node in Y:
                    # This node is on a back-door path
                    result.add(parent)
                    result |= visited
                    break

                # Continue BFS: go to neighbors (parents + children)
                for neighbor in dag.parents(node) | dag.children(node):
                    if neighbor not in visited:
                        queue.append(neighbor)

    return result


def _blocks_back_door(
    dag: CausalDAG, X: Set[str], Y: Set[str], Z: Set[str]
) -> bool:
    """Test whether Z blocks all back-door paths from X to Y."""
    # For each x ∈ X, each back-door path x ← ... → y must be blocked
    for x in X:
        for y in Y:
            for parent in dag.parents(x):
                # Check if there's an active back-door path from parent to y
                if not _path_blocked(dag, parent, y, Z | X, {x}):
                    return False
    return True


def _path_blocked(
    dag: CausalDAG, start: str, end: str, conditioning: Set[str],
    forbidden: Set[str],
) -> bool:
    """Check if all paths from start to end are blocked by conditioning."""
    # Simple BFS: if we can reach end through paths not passing through
    # forbidden nodes and where conditioning blocks appropriately
    visited: Set[Tuple[str, bool]] = set()  # (node, came_from_child)
    queue = [(start, False)]  # False = came from parent

    while queue:
        node, from_child = queue.pop()
        if (node, from_child) in visited or node in forbidden:
            continue
        visited.add((node, from_child))

        if node == end:
            return False  # Found unblocked path

        if from_child:
            # Came from child → node is a collider on this path
            # Collider path is active only if node is conditioned on
            if node in conditioning:
                # Continue to parents and children
                for child in dag.children(node):
                    if (child, False) not in visited:
                        queue.append((child, False))
            # Also: can continue to parents (since it's a collider being conditioned)
            if node in conditioning:
                for parent in dag.parents(node):
                    if (parent, True) not in visited:
                        queue.append((parent, True))
        else:
            # Came from parent → node is non-collider
            if node not in conditioning:
                # Path is open through non-collider
                for child in dag.children(node):
                    if (child, False) not in visited:
                        queue.append((child, False))
                for parent in dag.parents(node):
                    if (parent, True) not in visited:
                        queue.append((parent, True))

    return True


def descendants_union(dag: CausalDAG, nodes: Set[str]) -> Set[str]:
    """Union of descendants of all nodes in the set."""
    result: Set[str] = set()
    for n in nodes:
        result |= dag.descendants(n)
    return result


# ── front-door criterion ────────────────────────────────────────

def find_front_door_adjustment(
    dag: CausalDAG,
    treatment: str,
    outcome: str,
) -> Optional[Tuple[List[str], str]]:
    """
    Find front-door adjustment.

    A set M satisfies the front-door criterion relative to (X, Y) if:
      1. M intercepts all directed paths from X to Y
      2. There is no unblocked back-door path from X to M
      3. All back-door paths from M to Y are blocked by X

    The front-door formula:
      P(y | do(x)) = Σ_m P(m | x) Σ_x' P(y | x', m) P(x')
    """
    # Find candidate mediators: all nodes on directed paths X → ... → Y
    candidates = []
    for v in dag.variables:
        if v in (treatment, outcome):
            continue
        if (treatment in dag.ancestors(v) and
                v in dag.ancestors(outcome)):
            # v is on a directed path from X to Y
            # Check condition 2: no back-door path from X to v
            bd = _find_back_door_nodes(dag, {treatment}, {v})
            if not bd:
                candidates.append(v)

    if not candidates:
        return None

    # Use the first valid mediator
    M = candidates[0]

    # Check condition 3: back-door from M to Y blocked by {X}
    bd_MY = _find_back_door_nodes(dag, {M}, {outcome})
    bd_MY_blocked_by_X = all(
        dag.is_d_separated([m], [outcome], [treatment])
        for m in [M]
    )

    if not bd_MY_blocked_by_X:
        return None

    expr = (
        f"P({outcome} | do({treatment})) = "
        f"Σ_{{{M}}} P({M} | {treatment}) "
        f"Σ_{{{treatment}'}} P({outcome} | {treatment}', {M}) "
        f"P({treatment}')"
    )

    return ([M], expr)


# ── instrumental variable ───────────────────────────────────────

def check_instrument(
    dag: CausalDAG, instrument: str, treatment: str, outcome: str,
) -> IdentificationResult:
    """
    Check whether a variable is a valid instrument.

    Conditions for Z to be an instrument for X → Y:
      1. Z → X (relevance: Z affects X)
      2. Z ⊥ Y | X, U (exclusion: no direct effect on Y)
      3. Z ⊥ U (unconfounded: no common cause with outcome)
    """
    issues = []

    # Condition 1
    if treatment not in dag.children(instrument):
        issues.append(f"{instrument} does not affect {treatment}")

    # Condition 2 (graphical): no direct edge Z→Y and all Z→Y paths go through X
    if outcome in dag.children(instrument):
        issues.append(f"Direct edge {instrument}→{outcome} violates exclusion")

    # Check if there's any backdoor from Z to Y not through X
    bd = _find_back_door_nodes(dag, {instrument}, {outcome})
    if bd and not all(
        dag.is_d_separated([instrument], [outcome], [treatment])
    ):
        issues.append(f"Exclusion restriction may be violated")

    if issues:
        return IdentificationResult(
            False,
            method="IV",
            explanation="; ".join(issues),
        )

    return IdentificationResult(
        True,
        method="Instrumental Variable",
        adjustment_set=[],
        expression=(
            f"ATE = Cov({outcome}, {instrument}) / "
            f"Cov({treatment}, {instrument})"
        ),
        explanation=f"{instrument} is a valid instrument for {treatment}→{outcome}",
    )


# ── main identification entry point ─────────────────────────────

def identify_effect(
    dag: CausalDAG,
    treatment: Union[str, List[str]],
    outcome: Union[str, List[str]],
) -> IdentificationResult:
    """
    Try to identify P(outcome | do(treatment)) using available methods.

    Tries in order: back-door, front-door, IV.
    """
    if isinstance(treatment, str):
        treatment = [treatment]
    if isinstance(outcome, str):
        outcome = [outcome]

    explanations = []

    # 1. Check if treatment and outcome are d-separated (no effect at all)
    if dag.is_d_separated(treatment, outcome, []):
        return IdentificationResult(
            True,
            method="d-separation",
            expression=f"P({outcome} | do({treatment})) = P({outcome})",
            explanation="Treatment and outcome are marginally independent",
        )

    # 2. Back-door
    adj = find_back_door_adjustment(dag, treatment, outcome)
    if adj is not None:
        adj_str = ", ".join(adj) if adj else "∅ (no confounding)"
        return IdentificationResult(
            True,
            method="Back-door adjustment",
            adjustment_set=adj,
            expression=(
                f"P({outcome} | do({treatment})) = "
                f"Σ_{{{adj_str}}} P({outcome} | {treatment}, {adj_str}) "
                f"P({adj_str})"
            ),
            explanation=f"Adjusting for {adj_str} blocks all back-door paths",
        )

    # 3. Front-door (only for single treatment/outcome)
    if len(treatment) == 1 and len(outcome) == 1:
        fd = find_front_door_adjustment(dag, treatment[0], outcome[0])
        if fd is not None:
            mediators, expr = fd
            return IdentificationResult(
                True,
                method="Front-door adjustment",
                adjustment_set=mediators,
                expression=expr,
                explanation=f"Mediator(s) {mediators} satisfy front-door criterion",
            )

    # 4. do-calculus fallback — search for valid adjustment via Rules 1-3
    t_set = set(treatment)
    o_set = set(outcome)

    # Collect candidates: non-treatment, non-outcome, non-descendant of treatment
    t_descendants: Set[str] = set()
    for t in treatment:
        t_descendants |= dag.descendants(t)

    candidates = [v for v in dag.variables
                  if v not in t_set and v not in o_set and v not in t_descendants]

    # Try all subsets W as potential adjustment sets via Rule 1
    # Rule 1: P(y|do(x),z,w) = P(y|do(x),w) if Y ⊥ Z | X,W in G_{X̄}
    # This allows us to iteratively remove conditioning variables,
    # reducing P(y|do(x)) to Σ_w P(y|do(x),w)P(w) ≡ Σ_w P(y|x,w)P(w)
    from itertools import combinations

    dag_bar = dag.remove_outgoing(t_set)  # G_{X̄}: remove outgoing from X

    for size in range(len(candidates) + 1):
        for W in combinations(candidates, size):
            W_set = set(W)
            # Remaining variables not in W
            remaining = set(candidates) - W_set

            # Check if W blocks ALL remaining back-door paths:
            # Y ⊥ remaining | t_set, W in G_{X̄}
            if remaining and dag_bar.is_d_separated(o_set, remaining, t_set | W_set):
                # W is a valid adjustment set — all remaining variables
                # are conditionally independent of Y given X,W in the mutilated graph
                W_list = sorted(W_set)
                W_str = ", ".join(W_list) if W_list else "∅"
                return IdentificationResult(
                    True,
                    method="do-calculus (Rule 1 — generalized back-door)",
                    adjustment_set=W_list,
                    expression=(
                        f"P({outcome} | do({treatment})) = "
                        f"Σ_{{{W_str}}} P({outcome} | {treatment}, {W_str}) "
                        f"P({W_str})"
                    ),
                    explanation=(
                        f"do-calculus Rule 1: Y ⊥ {{{', '.join(sorted(remaining))}}} "
                        f"| {treatment}, {W_str} in G_{{X̄}}"
                    ),
                )

            # Edge case: empty remaining set means W already accounts for
            # all back-door variables — we've found the adjustment
            if not remaining:
                W_list = sorted(W_set)
                W_str = ", ".join(W_list) if W_list else "∅"
                return IdentificationResult(
                    True,
                    method="do-calculus (Rule 1 — exhaustive)",
                    adjustment_set=W_list,
                    expression=(
                        f"P({outcome} | do({treatment})) = "
                        f"Σ_{{{W_str}}} P({outcome} | {treatment}, {W_str}) "
                        f"P({W_str})"
                    ),
                    explanation=f"All candidate confounders accounted for via {W_str}",
                )

    # 5. Not identifiable by any available method
    return IdentificationResult(
        False,
        explanation=(
            "Effect not identifiable via back-door, front-door, IV, "
            "or do-calculus Rules 1-3. "
            "May require experimental data or additional assumptions."
        ),
    )


# ── do-calculus (rule-based engine) ─────────────────────────────

def do_calculus_rule1(
    dag: CausalDAG, Y: Set[str], X: Set[str], Z: Set[str], W: Set[str]
) -> bool:
    """
    Rule 1: Insertion/deletion of observations.

    P(y | do(x), z, w) = P(y | do(x), w)  if Y ⊥ Z | X, W  in DAG_{X̄}
    (X̄ = remove edges outgoing from X)
    """
    dag_bar = dag.remove_outgoing(X)
    return dag_bar.is_d_separated(Y, Z, X | W)


def do_calculus_rule2(
    dag: CausalDAG, Y: Set[str], X: Set[str], Z: Set[str], W: Set[str]
) -> bool:
    """
    Rule 2: Action/observation exchange.

    P(y | do(x), do(z), w) = P(y | do(x), z, w)  if Y ⊥ Z | X, W  in DAG_{X̄, Z̲}
    (remove edges outgoing from X, incoming to Z)
    """
    dag_modified = dag.remove_outgoing(X)
    # Also remove edges into Z
    # Create new DAG without edges into Z
    edges = []
    for u in dag_modified.variables:
        for v in dag_modified.children(u):
            if v not in Z:
                edges.append((u, v))
    dag_m = CausalDAG(dag_modified.variables, edges)
    return dag_m.is_d_separated(Y, Z, X | W)


def do_calculus_rule3(
    dag: CausalDAG, Y: Set[str], X: Set[str], Z: Set[str], W: Set[str]
) -> bool:
    """
    Rule 3: Insertion/deletion of actions.

    P(y | do(x), do(z), w) = P(y | do(x), w)  if Y ⊥ Z | X, W  in DAG_{X̄, Z(W)̄}
    where Z(W) = Z \ an(W)_{DAG_{X̄}}
    """
    dag_bar = dag.remove_outgoing(X)
    # an(W) in DAG_{X̄}
    an_W = set()
    for w_node in W:
        an_W |= dag_bar.ancestors(w_node)
    Z_trimmed = Z - an_W
    # Remove outgoing from Z_trimmed
    edges = []
    for u in dag_bar.variables:
        for v in dag_bar.children(u):
            if u not in Z_trimmed:
                edges.append((u, v))
    dag_mod = CausalDAG(dag_bar.variables, edges)
    return dag_mod.is_d_separated(Y, Z, X | W)


# ── tests ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test 1: Simple confounding (back-door)
    # Z → X, Z → Y, X → Y
    dag = CausalDAG(["Z", "X", "Y"], [("Z", "X"), ("Z", "Y"), ("X", "Y")])
    result = identify_effect(dag, "X", "Y")
    print("Test 1 — Confounded X→Y:")
    print(result)

    # Test 2: Front-door
    # X → M → Y, with unobserved U → X, U → Y
    # Represent U as latent (not in graph), so X and Y share hidden confounding
    # In the observed graph: X → M → Y (no Z variable)
    dag2 = CausalDAG(["X", "M", "Y"], [("X", "M"), ("M", "Y")])
    result2 = identify_effect(dag2, "X", "Y")
    print("\nTest 2 — Front-door:")
    print(result2)

    # Test 3: M-bias graph
    # U1→X, U1→Z, U2→Z, U2→Y, X→Y
    dag3 = CausalDAG(
        ["U1", "U2", "X", "Y", "Z"],
        [("U1", "X"), ("U1", "Z"), ("U2", "Z"), ("U2", "Y"), ("X", "Y")],
    )
    result3 = identify_effect(dag3, "X", "Y")
    print("\nTest 3 — M-bias:")
    print(result3)

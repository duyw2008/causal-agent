"""
Causal DAG — directed acyclic graph with causal semantics.

Each node is a variable.  Each directed edge X → Y means
"X is a direct cause of Y" (relative to the set of variables in the graph).

The DAG encodes conditional independence assertions via d-separation
and serves as the backbone for do-calculus identification.
"""

from __future__ import annotations
from collections import deque
from itertools import combinations, chain
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple


class CausalDAG:
    """A causal directed acyclic graph.

    Parameters
    ----------
    variables : iterable of str
        Names of all variables.
    edges : iterable of (str, str)
        Directed edges (parent → child).
    """

    def __init__(
        self,
        variables: Iterable[str],
        edges: Iterable[Tuple[str, str]] = (),
    ):
        self._vars: List[str] = list(variables)
        self._index: Dict[str, int] = {v: i for i, v in enumerate(self._vars)}
        # parents[v] = set of direct causes of v
        self._parents: Dict[str, Set[str]] = {v: set() for v in self._vars}
        # children[v] = set of variables v directly causes
        self._children: Dict[str, Set[str]] = {v: set() for v in self._vars}

        for u, v in edges:
            self.add_edge(u, v)

        self._validate_acyclic()

    # ── basic accessors ──────────────────────────────────────────

    @property
    def variables(self) -> List[str]:
        return list(self._vars)

    @property
    def nodes(self) -> List[str]:
        return self._vars

    def parents(self, v: str) -> Set[str]:
        return set(self._parents[v])

    def children(self, v: str) -> Set[str]:
        return set(self._children[v])

    def ancestors(self, v: str) -> Set[str]:
        """All ancestors of v (including indirect)."""
        result: Set[str] = set()
        queue = list(self._parents[v])
        while queue:
            p = queue.pop()
            if p not in result:
                result.add(p)
                queue.extend(self._parents[p] - result)
        return result

    def descendants(self, v: str) -> Set[str]:
        """All descendants of v."""
        result: Set[str] = set()
        queue = list(self._children[v])
        while queue:
            c = queue.pop()
            if c not in result:
                result.add(c)
                queue.extend(self._children[c] - result)
        return result

    def add_edge(self, u: str, v: str):
        """Add a directed edge u → v."""
        if u not in self._parents or v not in self._parents:
            raise ValueError(f"Unknown variable: {u if u not in self._parents else v}")
        self._parents[v].add(u)
        self._children[u].add(v)

    # ── graph properties ────────────────────────────────────────

    def _validate_acyclic(self):
        """Raise if the graph contains a cycle."""
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(v: str) -> bool:
            visited.add(v)
            rec_stack.add(v)
            for child in self._children[v]:
                if child not in visited:
                    if dfs(child):
                        return True
                elif child in rec_stack:
                    return True
            rec_stack.discard(v)
            return False

        for v in self._vars:
            if v not in visited:
                if dfs(v):
                    cycle = [x for x in rec_stack]
                    raise ValueError(f"Cycle detected: {' → '.join(cycle)}")

    def topological_order(self) -> List[str]:
        """Return variables in causal (topological) order."""
        in_degree = {v: len(self._parents[v]) for v in self._vars}
        queue = deque(v for v in self._vars if in_degree[v] == 0)
        order = []
        while queue:
            v = queue.popleft()
            order.append(v)
            for child in self._children[v]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return order

    # ── d-separation ────────────────────────────────────────────

    def is_d_separated(
        self, X: Iterable[str], Y: Iterable[str], Z: Iterable[str]
    ) -> bool:
        """
        Test whether X and Y are d-separated given Z.

        Uses the moralised-graph reachability criterion (linear in |V|+|E|).
        """
        X = set(X)
        Y = set(Y)
        Z = set(Z)

        # Step 1: find all ancestors of X ∪ Y ∪ Z
        an = set(chain.from_iterable(
            self.ancestors(v) for v in chain(X, Y, Z)
        ))
        an |= X | Y | Z

        # Step 2: moralise — connect unmarried parents, make undirected
        moral = {v: set() for v in an}
        for v in an:
            for child in self._children[v]:
                if child in an:
                    moral[v].add(child)
                    moral[child].add(v)
        # connect spouses (parents of a common child)
        for v in an:
            pa = [p for p in self._parents[v] if p in an]
            for a, b in combinations(pa, 2):
                moral[a].add(b)
                moral[b].add(a)

        # Step 3: remove Z nodes (and their incident edges) from moral graph
        # (equivalent to: BFS from X, cannot pass through Z unless Z is a collider descendant)

        # Simpler: use the standard algorithm through the ancestral graph.
        # We'll BFS on the original DAG with path rules.
        return self._dsep_traverse(X, Y, Z)

    def _dsep_traverse(
        self, X: Set[str], Y: Set[str], Z: Set[str]
    ) -> bool:
        """
        Standard d-separation via the moralised-graph test.

        1. Keep only nodes that are ancestors of X ∪ Y ∪ Z.
        2. Moralise: for each node, connect its parents pairwise,
           then make all edges undirected.
        3. Remove nodes in Z (and their edges).
        4. X and Y are d-separated given Z iff they are disconnected
           in the resulting undirected graph.
        """
        an_set: Set[str] = set()
        for node in X | Y | Z:
            an_set.add(node)
            an_set |= self.ancestors(node)

        # --- Step 2: moralise ---
        # adjacency: undirected edges in the moral graph
        adj: Dict[str, Set[str]] = {v: set() for v in an_set}

        for v in an_set:
            for child in self._children[v]:
                if child in an_set:
                    adj[v].add(child)
                    adj[child].add(v)
            # connect parents (spouses)
            parents_in_an = [p for p in self._parents[v] if p in an_set]
            for i in range(len(parents_in_an)):
                for j in range(i + 1, len(parents_in_an)):
                    a, b = parents_in_an[i], parents_in_an[j]
                    adj[a].add(b)
                    adj[b].add(a)

        # --- Step 3: remove Z ---
        for z in Z:
            if z in adj:
                for neighbour in adj[z]:
                    adj[neighbour].discard(z)
                del adj[z]

        # --- Step 4: check connectivity ---
        # BFS from any X node to any Y node
        if not (X & an_set) or not (Y & an_set):
            return True  # one side not in ancestral set → no path

        visited: Set[str] = set()
        queue: List[str] = [next(iter(X & an_set))]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            if node in Y:
                return False  # connected → NOT d-separated
            if node in adj:
                for nb in adj[node]:
                    if nb not in visited:
                        queue.append(nb)

        return True  # disconnected → d-separated

    # ── do-calculus helpers ─────────────────────────────────────

    def mutilate(self, X: Iterable[str]) -> CausalDAG:
        """Return a new DAG with all edges INTO X removed (do(X) operation)."""
        X = set(X)
        new_edges = []
        for v in self._vars:
            for child in self._children[v]:
                if child not in X:
                    new_edges.append((v, child))
        return CausalDAG(self._vars, new_edges)

    def remove_outgoing(self, X: Iterable[str]) -> CausalDAG:
        """Return DAG with edges OUT of X removed."""
        X = set(X)
        new_edges = []
        for v in self._vars:
            for child in self._children[v]:
                if v not in X:
                    new_edges.append((v, child))
        return CausalDAG(self._vars, new_edges)

    # ── display ─────────────────────────────────────────────────

    def __repr__(self) -> str:
        edges = [f"{u}→{v}" for u in self._vars
                 for v in self._children[u]]
        return f"CausalDAG({', '.join(edges) if edges else 'no edges'})"

    def to_mermaid(self) -> str:
        """Export to Mermaid flowchart syntax for rendering."""
        lines = ["graph LR"]
        for v in self._vars:
            for child in sorted(self._children[v]):
                lines.append(f"    {v} --> {child}")
        return "\n".join(lines)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"Causal DAG: {len(self._vars)} variables, "
                 f"{sum(len(c) for c in self._children.values())} edges"]
        for v in self.topological_order():
            pa = sorted(self._parents[v])
            ch = sorted(self._children[v])
            pa_str = f"  parents: [{', '.join(pa)}]" if pa else "  (exogenous)"
            ch_str = f"  children: [{', '.join(ch)}]" if ch else ""
            lines.append(f"  {v}:{pa_str}{ch_str}")
        return "\n".join(lines)


# ── quick tests ─────────────────────────────────────────────────
if __name__ == "__main__":
    # Chain: X → M → Y.  X ⊥ Y | M.
    dag1 = CausalDAG(["X", "M", "Y"], [("X", "M"), ("M", "Y")])
    assert dag1.is_d_separated(["X"], ["Y"], ["M"]), "chain should block"
    assert not dag1.is_d_separated(["X"], ["Y"], []), "chain open without M"

    # Fork: X ← Z → Y.  X ⊥ Y | Z.
    dag2 = CausalDAG(["X", "Y", "Z"], [("Z", "X"), ("Z", "Y")])
    assert dag2.is_d_separated(["X"], ["Y"], ["Z"]), "fork should block"

    # Collider: X → Z ← Y.  X ⊥ Y (marginally), X ̸⊥ Y | Z.
    dag3 = CausalDAG(["X", "Y", "Z"], [("X", "Z"), ("Y", "Z")])
    assert dag3.is_d_separated(["X"], ["Y"], []), "collider open margin"
    assert not dag3.is_d_separated(["X"], ["Y"], ["Z"]), "collider closed given Z"

    print("All DAG tests passed ✓")
    print(dag1.summary())

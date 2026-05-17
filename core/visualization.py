"""
DAG Visualization — render causal graphs in terminal and files.

Output formats:
  - ASCII art (terminal)
  - Mermaid markdown (GraphML-compatible)
  - DOT (graphviz)
  - PNG/SVG (requires graphviz: pip install graphviz)
"""

from __future__ import annotations
import subprocess
import tempfile
from pathlib import Path
from typing import Optional


def dag_to_ascii(dag) -> str:
    """
    Render a CausalDAG as ASCII art in the terminal.

    Uses a simple layered layout based on topological order.
    """
    order = dag.topological_order()
    n = len(order)
    if n == 0:
        return "(empty DAG)"

    # Assign layer to each node
    layers: dict = {}
    for v in order:
        parents = dag.parents(v)
        if not parents:
            layers[v] = 0
        else:
            layers[v] = max(layers.get(p, 0) for p in parents) + 1

    # Group by layer
    max_layer = max(layers.values()) if layers else 0
    layer_nodes = {i: [] for i in range(max_layer + 1)}
    for v, l in layers.items():
        layer_nodes[l].append(v)

    # Build ASCII
    lines = []
    max_width = 60

    for l in range(max_layer + 1):
        nodes = layer_nodes[l]
        if not nodes:
            continue

        # Position nodes evenly
        spacing = max_width // (len(nodes) + 1)
        line = [" "] * max_width

        for i, node in enumerate(nodes):
            pos = spacing * (i + 1)
            for j, ch in enumerate(node):
                if pos + j < max_width:
                    line[pos + j] = ch

        lines.append("".join(line).rstrip())

        # Draw edges to next layer
        if l < max_layer:
            edge_line = [" "] * max_width
            for node in nodes:
                pos = spacing * (layer_nodes[l].index(node) + 1)
                for child in dag.children(node):
                    if child in layer_nodes.get(l + 1, []):
                        child_pos = spacing * (layer_nodes[l + 1].index(child) + 1)
                        # Draw vertical line
                        x = (pos + child_pos) // 2
                        if 0 < x < max_width:
                            edge_line[x] = "│"
                        # Draw connector
                        for cx in range(min(pos, child_pos), max(pos, child_pos)):
                            if 0 < cx < max_width and edge_line[cx] == " ":
                                edge_line[cx] = "─"
            lines.append("".join(edge_line).rstrip())

    # Legend
    edges_list = []
    for v in order:
        for child in sorted(dag.children(v)):
            edges_list.append(f"{v}→{child}")

    result = "\n".join(lines)
    if edges_list:
        result += f"\n\n  Edges: {', '.join(edges_list)}"
    result += f"\n  Order: {' → '.join(order)}"

    return result


def pag_to_ascii(pag) -> str:
    """Render a Partial Ancestral Graph (PAG) as ASCII art.

    Shows PAG-specific edge types that dag_to_ascii loses:
      →   (directed)     A causes B
      ◦→  (ancestral)    A is ancestor of B (possibly indirect)
      ◦—◦ (undetermined) unknown relationship
      ↔   (confounded)   hidden common cause
    """
    from .discovery import PAG, PAGEdge
    if not isinstance(pag, PAG):
        return dag_to_ascii(pag)

    n = pag.n
    names = pag.var_names

    # Count edges
    n_edges = 0
    seen = set()
    for i in range(n):
        for j, edge in pag._adj[i].items():
            if (j, i) not in seen:
                seen.add((i, j))
                n_edges += 1

    lines = []
    lines.append(f"PAG: {n} variables, {n_edges} edges")
    lines.append("  Edge legend: → directed  ◦→ ancestral  ◦—◦ undetermined  ↔ confounded")
    lines.append("")

    mark_map = {0: "—", 1: "→", 2: "◦"}

    for (i, j) in sorted(seen):
        edge = pag._adj[i].get(j)
        if edge is None:
            continue
        left = mark_map.get(edge.mark_u, "?")
        right = mark_map.get(edge.mark_v, "?")
        symbol = f"{left}{right}"

        if edge.is_directed:
            etype = "directed"
        elif edge.is_bidirected:
            etype = "confounded (↔)"
        elif edge.is_undetermined:
            etype = "undetermined (◦—◦ or ◦→)"
        else:
            etype = "undirected"

        lines.append(f"  {names[i]} {symbol} {names[j]:10s}  [{etype}]")

    lines.append("")
    lines.append("  Adjacency:")
    for i in range(n):
        adj_list = [names[j] for j in pag._adj[i]]
        lines.append(f"    {names[i]}: {{{', '.join(adj_list)}}}")

    return "\n".join(lines)


def dag_to_dot(dag, highlight_treatment=None, highlight_outcome=None,
               highlight_adjustment=None) -> str:
    """
    Export CausalDAG to Graphviz DOT format.

    Parameters
    ----------
    highlight_treatment : str or None
        Node to highlight in red.
    highlight_outcome : str or None
        Node to highlight in blue.
    highlight_adjustment : list or None
        Nodes to outline in green (adjustment set).
    """
    lines = ["digraph G {",
             "  rankdir=LR;",
             "  node [shape=ellipse, style=filled, fillcolor=white, fontname=Helvetica];",
             "  edge [fontname=Helvetica];"]

    adj_set = set(highlight_adjustment or [])

    for v in dag.variables:
        attrs = []
        if v == highlight_treatment:
            attrs.append('fillcolor="#FEE2E2"')      # red tint
            attrs.append('fontcolor="#991B1B"')
        elif v == highlight_outcome:
            attrs.append('fillcolor="#DBEAFE"')      # blue tint
            attrs.append('fontcolor="#1E40AF"')
        elif v in adj_set:
            attrs.append('fillcolor="#D1FAE5"')      # green tint
            attrs.append('fontcolor="#065F46"')
            attrs.append('penwidth=2')

        attr_str = ", ".join(attrs)
        lines.append(f'  "{v}" [{attr_str}];')

    for v in dag.variables:
        for child in sorted(dag.children(v)):
            lines.append(f'  "{v}" -> "{child}";')

    lines.append("}")
    return "\n".join(lines)


def dag_to_mermaid(dag) -> str:
    """Export to Mermaid flowchart syntax."""
    return dag.to_mermaid()


def render_dag(dag, filepath: str, fmt: str = "png",
               highlight_treatment=None, highlight_outcome=None,
               highlight_adjustment=None) -> Optional[str]:
    """
    Render DAG to PNG/SVG/PDF using graphviz.

    Requires: pip install graphviz (or system graphviz)

    Returns the output filepath, or None if graphviz is unavailable.
    """
    dot = dag_to_dot(dag, highlight_treatment, highlight_outcome,
                     highlight_adjustment)

    # Try graphviz Python library first
    try:
        import graphviz
        g = graphviz.Source(dot, format=fmt)
        out_path = str(Path(filepath).with_suffix(f".{fmt}"))
        g.render(filename=str(Path(filepath).with_suffix("")),
                 format=fmt, cleanup=True)
        return out_path
    except ImportError:
        pass

    # Try system 'dot' command
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as f:
            f.write(dot)
            dot_path = f.name

        out_path = str(Path(filepath).with_suffix(f".{fmt}"))
        subprocess.run(
            ["dot", f"-T{fmt}", dot_path, "-o", out_path],
            check=True, capture_output=True, timeout=10,
        )
        Path(dot_path).unlink(missing_ok=True)
        return out_path
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


# ═══════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/duyw/causal_agent')
    from core.graph import CausalDAG

    print("=" * 55)
    print("  DAG VISUALIZATION TESTS")
    print("=" * 55)

    # Simpson's paradox DAG
    dag = CausalDAG(["G", "D", "R"], [("G", "D"), ("G", "R"), ("D", "R")])

    print("\n── ASCII Art ──")
    print(dag_to_ascii(dag))

    print("\n── DOT (highlighted) ──")
    print(dag_to_dot(dag, highlight_treatment="D", highlight_outcome="R",
                     highlight_adjustment=["G"]))

    print("\n── Mermaid ──")
    print(dag_to_mermaid(dag))

    # Try rendering to PNG
    png_path = render_dag(dag, "/tmp/test_dag", fmt="png",
                          highlight_treatment="D", highlight_outcome="R",
                          highlight_adjustment=["G"])
    if png_path:
        print(f"\n── PNG rendered ──")
        print(f"  {png_path}")
    else:
        print(f"\n── PNG ──")
        print(f"  (graphviz not available; install with: pip install graphviz)")

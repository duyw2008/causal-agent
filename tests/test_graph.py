"""Tests for CausalDAG and d-separation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.graph import CausalDAG


class TestCausalDAG:
    def test_chain_dseparation(self):
        """X→M→Y: X ⊥ Y | M, X not⊥ Y unconditionally."""
        dag = CausalDAG(["X", "M", "Y"], [("X", "M"), ("M", "Y")])
        assert dag.is_d_separated(["X"], ["Y"], ["M"])
        assert not dag.is_d_separated(["X"], ["Y"], [])

    def test_fork_dseparation(self):
        """X←Z→Y: X ⊥ Y | Z."""
        dag = CausalDAG(["X", "Y", "Z"], [("Z", "X"), ("Z", "Y")])
        assert dag.is_d_separated(["X"], ["Y"], ["Z"])
        assert not dag.is_d_separated(["X"], ["Y"], [])

    def test_collider_dseparation(self):
        """X→Z←Y: X ⊥ Y unconditionally, X not⊥ Y | Z."""
        dag = CausalDAG(["X", "Y", "Z"], [("X", "Z"), ("Y", "Z")])
        assert dag.is_d_separated(["X"], ["Y"], [])
        assert not dag.is_d_separated(["X"], ["Y"], ["Z"])

    def test_topological_order(self):
        """Topological order respects edge directions."""
        dag = CausalDAG(["A", "B", "C"], [("A", "B"), ("B", "C")])
        order = dag.topological_order()
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_parents_children(self):
        dag = CausalDAG(["X", "Y", "Z"], [("X", "Y"), ("X", "Z")])
        assert dag.parents("X") == set()
        assert dag.children("X") == {"Y", "Z"}
        assert dag.parents("Y") == {"X"}
        assert dag.children("Y") == set()

    def test_ancestors_descendants(self):
        dag = CausalDAG(["A", "B", "C", "D"],
                        [("A", "B"), ("B", "C"), ("C", "D")])
        assert dag.ancestors("D") == {"A", "B", "C"}
        assert dag.descendants("A") == {"B", "C", "D"}

    def test_cycle_detection(self):
        """Adding a cycle should raise ValueError."""
        try:
            CausalDAG(["A", "B", "C"], [("A", "B"), ("B", "C"), ("C", "A")])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_mutilate(self):
        """do(X) removes incoming edges to X."""
        dag = CausalDAG(["Z", "X", "Y"], [("Z", "X"), ("X", "Y")])
        mut = dag.mutilate(["X"])
        assert "Z" not in mut.parents("X")
        assert mut.children("X") == {"Y"}

    def test_remove_outgoing(self):
        dag = CausalDAG(["X", "Y", "Z"], [("X", "Y"), ("Y", "Z")])
        rm = dag.remove_outgoing(["X"])
        assert rm.children("X") == set()
        assert rm.children("Y") == {"Z"}

    def test_mermaid_export(self):
        dag = CausalDAG(["X", "Y"], [("X", "Y")])
        mmd = dag.to_mermaid()
        assert "graph LR" in mmd
        assert "X --> Y" in mmd

    def test_simpson_dag_dsep(self):
        """Gender→Drug, Gender→Recovery, Drug→Recovery.
        Drug and Recovery are NOT d-separated unconditionally.
        They ARE d-separated given Gender if there's no direct edge,
        but here Drug→Recovery exists so they are never d-separated."""
        dag = CausalDAG(["G", "D", "R"], [("G", "D"), ("G", "R"), ("D", "R")])
        assert not dag.is_d_separated(["D"], ["R"], [])
        assert not dag.is_d_separated(["D"], ["R"], ["G"])

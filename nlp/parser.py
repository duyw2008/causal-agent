"""
Natural language → causal model parser.

Translates human-readable scenario descriptions into:
  - CausalDAG (variables + edges)
  - SCM (structural equations, for simple cases)
  - Query (target + intervention, when applicable)

Supports two modes:
  1. Rule-based templates for common patterns
  2. LLM-assisted parsing (when an LLM is available)
"""

from __future__ import annotations
import json
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from core.graph import CausalDAG
from core.scm import SCM, linear_scm, linear_eq, StructuralEquation


# ── rule-based extraction ───────────────────────────────────────

def extract_variables(text: str) -> List[str]:
    """Extract variable names from text.

    Looks for:
      - Capitalised words / acronyms (X, Y, Z, SES, BMI, etc.)
      - Explicit declarations like "variables: X, Y, Z"
      - Causal statements like "X causes Y"
    """
    # Explicit declaration
    m = re.search(
        r'variables?\s*[:：]\s*([A-Za-z_][A-Za-z0-9_]*(?:\s*[,，]\s*[A-Za-z_][A-Za-z0-9_]*)*)',
        text, re.IGNORECASE,
    )
    if m:
        return [v.strip() for v in re.split(r'[,，]+', m.group(1))]

    # Find causal statements
    causal_pattern = re.findall(
        r'\b([A-Z][A-Za-z0-9_]{0,20})\s+(?:causes?|affects?|influences?|→|->)\s+'
        r'\b([A-Z][A-Za-z0-9_]{0,20})',
        text,
    )
    if causal_pattern:
        vars_set = set()
        for u, v in causal_pattern:
            vars_set.add(u)
            vars_set.add(v)
        return sorted(vars_set)

    # Fallback: all uppercase words
    words = re.findall(r'\b[A-Z][A-Za-z0-9_]{1,20}\b', text)
    if words:
        seen = []
        for w in words:
            if w not in seen and w.lower() not in {
                'the', 'a', 'an', 'is', 'are', 'was', 'were', 'if',
                'we', 'in', 'on', 'at', 'to', 'of', 'for', 'and', 'or',
                'not', 'it', 'be', 'has', 'have', 'do', 'does', 'this',
                'that', 'with', 'from',
            }:
                seen.append(w)
        if 2 <= len(seen) <= 20:
            return seen

    return []


def extract_edges(text: str, variables: List[str]) -> List[Tuple[str, str]]:
    """Extract causal edges from text.

    Patterns:
      - "X causes Y"
      - "X affects Y"
      - "X influences Y"
      - "X → Y"
      - "X -> Y"
    """
    edges = []
    var_set = set(variables)

    patterns = [
        r'\b([A-Z][A-Za-z0-9_]*)\s+(?:causes?|affects?|influences?|leads?\s+to)\s+'
        r'\b([A-Z][A-Za-z0-9_]*)',
        r'\b([A-Z][A-Za-z0-9_]*)\s*[→]\s*\b([A-Z][A-Za-z0-9_]*)',
        r'\b([A-Z][A-Za-z0-9_]*)\s*->\s*\b([A-Z][A-Za-z0-9_]*)',
    ]

    for pat in patterns:
        for u, v in re.findall(pat, text, re.IGNORECASE):
            if u in var_set and v in var_set and u != v:
                # Prioritise: prefer the exact case from variables
                u_canon = next((x for x in variables if x.lower() == u.lower()), u)
                v_canon = next((x for x in variables if x.lower() == v.lower()), v)
                if (u_canon, v_canon) not in edges:
                    edges.append((u_canon, v_canon))

    return edges


def extract_numbers(text: str) -> Dict[str, float]:
    """Extract numeric coefficients like 'X = 0.5*Parent + ...'."""
    coeffs: Dict[str, float] = {}
    # Pattern: "variable = number * parent"
    # Or: "coefficient of X on Y is 0.5"
    pat1 = re.findall(
        r'\b([A-Z][A-Za-z0-9_]*)\s*=\s*([\d.]+)\s*\*\s*\b([A-Z][A-Za-z0-9_]*)',
        text,
    )
    for var, coef, parent in pat1:
        coeffs[f"{parent}→{var}"] = float(coef)

    # Pattern: "X affects Y by 0.5"
    pat2 = re.findall(
        r'\b([A-Z][A-Za-z0-9_]*)\s+(?:affects?|changes?)\s+'
        r'\b([A-Z][A-Za-z0-9_]*)\s+by\s+([\d.]+)',
        text,
    )
    for u, v, c in pat2:
        coeffs[f"{u}→{v}"] = float(c)

    return coeffs


def extract_query(text: str) -> Optional[Dict[str, Any]]:
    """Extract causal query from text.

    Patterns:
      - "What is the effect of X on Y?"
      - "What happens to Y if we set X to 1?"
      - "do(X=1)"
      - "intervene X=1"
    """
    query: Dict[str, Any] = {}

    # Intervention pattern
    intv_pat = re.findall(
        r'(?:do|set|intervene|intervention)\s*[\[(]?\s*'
        r'([A-Z][A-Za-z0-9_]*)\s*=\s*([\d.]+)',
        text, re.IGNORECASE,
    )
    if intv_pat:
        query["intervention"] = {v: float(val) for v, val in intv_pat}

    # Target (outcome) pattern
    outcome_pat = re.findall(
        r'(?:effect\s+(?:of\s+)?|outcome|target)\s+'
        r'(?:is\s+)?([A-Z][A-Za-z0-9_]*)',
        text, re.IGNORECASE,
    )
    if outcome_pat:
        query["outcome"] = outcome_pat[0]

    # Treatment pattern
    treatment_pat = re.findall(
        r'(?:effect\s+of|treatment|exposure)\s+'
        r'(?:is\s+)?([A-Z][A-Za-z0-9_]*)',
        text, re.IGNORECASE,
    )
    if treatment_pat:
        query["treatment"] = treatment_pat[0]

    return query if query else None


# ── main parser ─────────────────────────────────────────────────

class CausalParser:
    """Parse a natural language description into a causal model."""

    def __init__(self, text: str):
        self.text = text
        self.variables: List[str] = []
        self.edges: List[Tuple[str, str]] = []
        self.coefficients: Dict[str, float] = {}
        self.query: Optional[Dict[str, Any]] = None
        self._parse()

    def _parse(self):
        self.variables = extract_variables(self.text)
        if not self.variables:
            raise ValueError(
                "Could not extract variables from text. "
                "Try naming them explicitly, e.g. 'variables: X, Y, Z'."
            )
        self.edges = extract_edges(self.text, self.variables)
        self.coefficients = extract_numbers(self.text)
        self.query = extract_query(self.text)

    def build_dag(self) -> CausalDAG:
        """Build a CausalDAG from the parsed description."""
        if not self.edges:
            # If no edges extracted, assume all are observed but independent
            return CausalDAG(self.variables, [])
        return CausalDAG(self.variables, self.edges)

    def build_scm(self) -> Optional[SCM]:
        """Attempt to build an SCM if coefficients are available."""
        dag = self.build_dag()
        if not self.coefficients:
            return None

        # Convert coefficient dict to per-variable format
        coeffs: Dict[str, Dict[str, float]] = {}
        for edge, val in self.coefficients.items():
            if "→" in edge:
                u, v = edge.split("→")
                if v not in coeffs:
                    coeffs[v] = {}
                coeffs[v][u] = val

        return linear_scm(dag, coeffs, noise_std=0.1)

    def summary(self) -> str:
        lines = [f"Parsed causal model from description:", f""]
        lines.append(f"  Variables ({len(self.variables)}): {', '.join(self.variables)}")
        if self.edges:
            edge_str = ", ".join(f"{u}→{v}" for u, v in self.edges)
            lines.append(f"  Edges ({len(self.edges)}): {edge_str}")
        else:
            lines.append(f"  Edges: (none detected)")
        if self.coefficients:
            lines.append(f"  Coefficients: {self.coefficients}")
        if self.query:
            lines.append(f"  Query: {self.query}")
        return "\n".join(lines)


# ── template library ────────────────────────────────────────────

TEMPLATES = {
    "smoking_lung_cancer": {
        "description": (
            "Smoking (S) causes both Tar (T) in lungs and Lung Cancer (C). "
            "Tar also causes Lung Cancer. There may be a genetic factor (G) "
            "that affects both Smoking and Lung Cancer."
        ),
        "variables": ["G", "S", "T", "C"],
        "edges": [("G", "S"), ("G", "C"), ("S", "T"), ("T", "C"), ("S", "C")],
    },
    "simpsons_paradox": {
        "description": (
            "A drug (D) affects recovery (R). But gender (G) affects both "
            "whether the drug is prescribed and the recovery rate."
        ),
        "variables": ["G", "D", "R"],
        "edges": [("G", "D"), ("G", "R"), ("D", "R")],
    },
    "education_income": {
        "description": (
            "Education (E) affects Income (I). Parental SES (S) affects "
            "both Education and Income."
        ),
        "variables": ["S", "E", "I"],
        "edges": [("S", "E"), ("S", "I"), ("E", "I")],
    },
    "front_door_example": {
        "description": (
            "Smoking (X) causes Tar (M) which causes Cancer (Y). "
            "There is an unobserved confounder U affecting both X and Y. "
            "(U is latent — not in the observed model)"
        ),
        "variables": ["X", "M", "Y"],
        "edges": [("X", "M"), ("M", "Y")],
    },
    "m_bias": {
        "description": (
            "U1 affects both X and Z. U2 affects both Z and Y. "
            "X causes Y. (U1, U2 are latent)"
        ),
        "variables": ["X", "Y", "Z"],
        "edges": [("X", "Y")],
        "note": "Z is a collider — conditioning on Z opens a spurious path",
    },
}


def load_template(name: str) -> Tuple[str, CausalDAG]:
    """Load a pre-built causal scenario by name."""
    if name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise ValueError(f"Unknown template '{name}'. Available: {available}")
    t = TEMPLATES[name]
    dag = CausalDAG(t["variables"], t["edges"])
    return t["description"], dag


# ── tests ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test rule-based parsing
    text = (
        "variables: Smoking, Tar, LungCancer. "
        "Smoking causes Tar. Tar causes LungCancer. "
        "Smoking affects LungCancer by 1.5. "
        "What is the effect of Smoking on LungCancer?"
    )
    parser = CausalParser(text)
    print(parser.summary())
    dag = parser.build_dag()
    print(dag.summary())
    scm = parser.build_scm()
    if scm:
        print(scm.summary())

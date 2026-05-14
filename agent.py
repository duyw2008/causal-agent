#!/usr/bin/env python3
"""
Causal Inference Agent — interactive reasoning engine.

Usage:
    python agent.py                          # interactive mode
    python agent.py "Does smoking cause cancer?"  # single query
    python agent.py --demo                  # run demo scenarios
"""

from __future__ import annotations
import sys
from typing import Any, Dict, List, Optional

from core.graph import CausalDAG
from core.scm import SCM, linear_scm, StructuralEquation
from core.identification import (
    identify_effect, IdentificationResult,
    find_back_door_adjustment, check_instrument,
)
from core.discovery import pc_algorithm, ges_algorithm, fci_algorithm, bootstrap_edge_confidence
from core.estimation import estimate_effect, CausalEstimate
from core.modern import (
    estimate_ate_dml, estimate_cate_slearner, estimate_cate_tlearner,
    estimate_cate_xlearner, estimate_cate_forest, estimate_ate_dowhy,
)
from core.sensitivity import full_sensitivity_report
from core.visualization import dag_to_ascii, dag_to_dot, render_dag
from nlp.parser import CausalParser, load_template
import numpy as np


# ── colour output ───────────────────────────────────────────────
class Style:
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    RESET = "\033[0m"


def green(s): return f"{Style.GREEN}{s}{Style.RESET}"
def cyan(s): return f"{Style.CYAN}{s}{Style.RESET}"
def yellow(s): return f"{Style.YELLOW}{s}{Style.RESET}"
def red(s): return f"{Style.RED}{s}{Style.RESET}"
def bold(s): return f"{Style.BOLD}{s}{Style.RESET}"


# ── agent core ──────────────────────────────────────────────────

class CausalAgent:
    """Reasoning agent combining causal graph, SCM, and inference."""

    def __init__(self):
        self.dag: Optional[CausalDAG] = None
        self.scm: Optional[SCM] = None
        self.data: Optional[np.ndarray] = None
        self.history: List[Dict[str, Any]] = []

    # ── scenario ingestion ───────────────────────────────────────

    def load_scenario(self, description: str) -> str:
        """Parse a scenario description and build the causal model."""
        lines = []

        # Try template first
        if description.strip().lower() in {
            "smoking", "simpson", "education", "frontdoor", "mbias",
            "smoking_lung_cancer", "simpsons_paradox", "education_income",
            "front_door_example", "m_bias",
        }:
            name_map = {
                "smoking": "smoking_lung_cancer",
                "simpson": "simpsons_paradox",
                "education": "education_income",
                "frontdoor": "front_door_example",
                "mbias": "m_bias",
            }
            name = name_map.get(description.strip().lower(), description.strip().lower())
            desc, self.dag = load_template(name)
            lines.append(green(f"Loaded template: {name}"))
            lines.append(f"  {desc}")
        else:
            try:
                parser = CausalParser(description)
                self.dag = parser.build_dag()
                self.scm = parser.build_scm()
                lines.append(green("Parsed scenario:"))
                lines.append(f"  Variables: {', '.join(self.dag.variables)}")
                if parser.edges:
                    edges_str = ", ".join(f"{u}→{v}" for u, v in parser.edges)
                    lines.append(f"  Edges: {edges_str}")
                if parser.query:
                    lines.append(f"  Query: {parser.query}")
            except ValueError as e:
                lines.append(red(f"Parse error: {e}"))
                return "\n".join(lines)

        lines.append("")
        lines.append(cyan("Causal DAG:"))
        lines.append(self.dag.summary())
        self.history.append({"type": "scenario", "description": description})
        return "\n".join(lines)

    # ── identification ──────────────────────────────────────────

    def ask_effect(self, treatment: str, outcome: str,
                   data: Optional[np.ndarray] = None,
                   method: str = "auto") -> str:
        """Ask: what is the causal effect of treatment on outcome?"""
        if self.dag is None:
            return red("No scenario loaded. Use 'load <description>' first.")

        if treatment not in self.dag.variables:
            return red(f"Unknown variable: {treatment}")
        if outcome not in self.dag.variables:
            return red(f"Unknown variable: {outcome}")

        lines = []
        lines.append(cyan(f"Query: P({outcome} | do({treatment}))"))
        lines.append("")

        # Check if there's a direct path
        if outcome in self.dag.descendants(treatment):
            lines.append(f"  {treatment} → ... → {outcome} — causal path exists")
        else:
            lines.append(f"  No causal path from {treatment} to {outcome} — ATE = 0")
            return "\n".join(lines)

        # Identification
        result = identify_effect(self.dag, treatment, outcome)
        lines.append(f"  Method: {result.method}")
        if result.adjustment_set:
            lines.append(f"  Adjustment set: {{{', '.join(result.adjustment_set)}}}")
        lines.append("")

        # Estimation (if data available)
        if data is not None and result.adjustment_set is not None:
            adj = result.adjustment_set
            if method in ("dml",):
                est = estimate_ate_dml(data, self.dag.variables, treatment, outcome, adj, n_folds=5)
            elif method in ("cate_s", "cate_t", "cate_x", "cate_forest"):
                cate_fn = {"cate_s": estimate_cate_slearner, "cate_t": estimate_cate_tlearner,
                           "cate_x": estimate_cate_xlearner, "cate_forest": estimate_cate_forest}[method]
                cate_est = cate_fn(data, self.dag.variables, treatment, outcome, adj)
                lines.append(bold(f"CATE Analysis ({cate_est.method}):"))
                lines.append(cate_est.summary())
                return "\n".join(lines)
            else:
                est = estimate_effect(
                    data, self.dag.variables, treatment, outcome,
                    adj, method=method,
                )
            lines.append(bold("Causal Effect Estimate:"))
            lines.append(f"  ATE = {est.ate:.4f}  (SE = {est.std_error:.4f})")
            lines.append(f"  95% CI = [{est.ci_lower:.4f}, {est.ci_upper:.4f}]")
            if est.is_significant():
                lines.append(green(f"  Statistically significant ✓"))
            else:
                lines.append(yellow(f"  Not statistically significant"))
            if est.warnings:
                for w in est.warnings:
                    lines.append(yellow(f"  ⚠ {w}"))

            # Sensitivity
            lines.append("")
            sens = full_sensitivity_report(est.ate, est.std_error)
            lines.append(sens)
        else:
            lines.append(yellow(
                "  (No data loaded for numerical estimation. "
                "Use 'load_data <file.csv>' or provide coefficients.)"
            ))

        return "\n".join(lines)

    # ── what-if (counterfactual / intervention) ──────────────────

    def what_if(
        self,
        intervention: Dict[str, float],
        target: str,
        observed: Optional[Dict[str, float]] = None,
    ) -> str:
        """What if we set X = x? Predict the value of target."""
        lines = []

        if observed:
            # Counterfactual
            if self.scm is None:
                return red("Counterfactual requires an SCM with structural equations. "
                           "Try loading a scenario with numeric coefficients.")
            cf = self.scm.counterfactual(observed, intervention, target)
            lines.append(cyan("Counterfactual inference:"))
            lines.append(f"  Observed: {observed}")
            lines.append(f"  Intervention: do({_format_intervention(intervention)})")
            lines.append(f"  → {target} = {cf:.4f}")
        else:
            # Interventional (population-level)
            if self.scm:
                intv_scm = self.scm.intervene(intervention)
                data = intv_scm.sample(10000, seed=42)
                mean_val = data[target].mean()
                lines.append(cyan("Interventional prediction:"))
                lines.append(f"  Intervention: do({_format_intervention(intervention)})")
                lines.append(f"  → E[{target}] = {mean_val:.4f}")
            else:
                lines.append(yellow(
                    "No SCM available. Only identification (not estimation) "
                    "is provided. Load a scenario with coefficients for predictions."
                ))

        return "\n".join(lines)

    # ── data loading ───────────────────────────────────────────

    def load_data(self, filepath: str) -> str:
        """Load CSV data for estimation."""
        try:
            self.data = np.genfromtxt(filepath, delimiter=',', skip_header=1,
                                      dtype=float)
            with open(filepath) as f:
                header = f.readline().strip().split(',')
            n_samples, n_vars = self.data.shape
            return green(
                f"Loaded {n_samples} samples, {n_vars} variables: "
                f"{', '.join(header)}"
            )
        except Exception as e:
            return red(f"Error loading data: {e}")

    # ── explain ─────────────────────────────────────────────────

    def explain(self, topic: str = "") -> str:
        """Explain the current model or a concept."""
        if not topic and self.dag:
            return (
                f"Current model:\n{self.dag.summary()}\n\n"
                f"D-separation example:\n"
                f"  Topological order: {self.dag.topological_order()}"
            )

        glossary = {
            "backdoor": (
                "Back-door criterion: A set Z blocks all back-door paths "
                "between X and Y if (1) no node in Z is a descendant of X, "
                "and (2) Z d-separates X from Y in the mutilated graph. "
                "Adjusting for Z removes confounding."
            ),
            "frontdoor": (
                "Front-door criterion: A mediator M that intercepts all "
                "causal paths from X to Y, with no back-door paths from X to M "
                "and all back-door paths from M to Y blocked by X."
            ),
            "dseparation": (
                "D-separation: X and Y are d-separated given Z if Z blocks "
                "every path between them. Paths are blocked by non-colliders "
                "in Z and unblocked by colliders in Z (or their descendants)."
            ),
            "do": (
                "do(X=x): Pearl's do-operator represents an intervention that "
                "sets X to x, removing all inbound edges to X. "
                "P(Y|do(X)) ≠ P(Y|X) when there is confounding."
            ),
            "scm": (
                "Structural Causal Model (SCM): M = ⟨U, V, F, P(u)⟩ where "
                "each endogenous variable v ∈ V is determined by a structural "
                "equation v = f(pa(v), u_v). Interventions replace equations; "
                "counterfactuals involve abduction-action-prediction."
            ),
        }

        t = topic.lower().replace(" ", "").replace("-", "")
        for key, explanation in glossary.items():
            if key in t:
                return f"{bold(key)}: {explanation}"

        return (
            f"Available topics: {', '.join(glossary.keys())}\n"
            f"Or ask about the current model."
        )

    # ── causal discovery ───────────────────────────────────────

    def discover(self, filepath: str, method: str = "pc",
                 alpha: float = 0.05, bootstrap: int = 0) -> str:
        """Discover causal structure from CSV data.

        Methods: pc, fci, ges
        Set bootstrap > 0 for edge confidence estimation.
        """
        lines = []
        try:
            data = np.genfromtxt(filepath, delimiter=',', skip_header=0,
                                 dtype=float, invalid_raise=False)
            # Check for header row
            with open(filepath) as f:
                first = f.readline().strip()
            try:
                [float(x) for x in first.split(',')]
                var_names = [f"V{i}" for i in range(data.shape[1])]
            except ValueError:
                var_names = [x.strip() for x in first.split(',')]
                data = np.genfromtxt(filepath, delimiter=',', skip_header=1,
                                     dtype=float)

            n_samples, n_vars = data.shape
            lines.append(green(f"Loaded {n_samples} samples, {n_vars} variables"))
            lines.append(f"  Variables: {', '.join(var_names)}")

            if method == "pc":
                lines.append(cyan(f"Running PC algorithm (alpha={alpha})..."))
                self.dag = pc_algorithm(
                    data, var_names, alpha=alpha, verbose=True,
                )
            elif method == "fci":
                lines.append(cyan(f"Running FCI algorithm (alpha={alpha})..."))
                pag = fci_algorithm(
                    data, var_names, alpha=alpha, verbose=True,
                )
                lines.append("")
                lines.append(cyan("FCI Partial Ancestral Graph:"))
                lines.append(pag.summary())
                self.dag = pag.to_dag()
                if not self.dag or not any(
                        self.dag.children(v) for v in self.dag.variables):
                    lines.append(yellow(
                        "  (No edges could be unambiguously directed. "
                        "Try with more variables or experimental data.)"
                    ))
            elif method == "ges":
                lines.append(cyan("Running GES algorithm..."))
                self.dag = ges_algorithm(data, var_names, verbose=True)
            else:
                return red(f"Unknown method: {method}. Use 'pc' or 'ges'.")

            lines.append("")
            lines.append(cyan("Discovered causal structure:"))
            lines.append(self.dag.summary())

            # Bootstrap
            if bootstrap > 0:
                lines.append("")
                lines.append(cyan(f"Bootstrap edge confidence (n={bootstrap}):"))
                conf = bootstrap_edge_confidence(
                    data, var_names, method=method,
                    n_bootstrap=bootstrap, alpha=alpha, verbose=True,
                )
                for (u, v), c in sorted(conf.items(), key=lambda x: -x[1]):
                    bar = "█" * int(c * 20) + "░" * (20 - int(c * 20))
                    lines.append(f"  {u}→{v}: {c:.2f}  {bar}")

            self.history.append({
                "type": "discovery", "file": filepath, "method": method,
            })
        except FileNotFoundError:
            return red(f"File not found: {filepath}")
        except Exception as e:
            return red(f"Discovery error: {e}")

        return "\n".join(lines)


def _format_intervention(d: Dict[str, float]) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items())


# ── demo ────────────────────────────────────────────────────────

def run_demo():
    agent = CausalAgent()
    print(bold("=" * 60))
    print(bold("  Causal Inference Agent — Demo"))
    print(bold("=" * 60))

    # Demo 1: Simpson's paradox
    print(f"\n{cyan('━' * 60)}")
    print(cyan("Demo 1: Simpson's Paradox"))
    print(cyan("━" * 60))
    print(agent.load_scenario("simpson"))
    print(f"\n{agent.explain('backdoor')}")
    print(f"\n{agent.ask_effect('D', 'R')}")

    # Demo 2: Front-door
    print(f"\n{cyan('━' * 60)}")
    print(cyan("Demo 2: Front-door Adjustment"))
    print(cyan("━" * 60))
    print(agent.load_scenario("frontdoor"))
    print(f"\n{agent.ask_effect('X', 'Y')}")

    # Demo 3: Smoking with coefficients
    print(f"\n{cyan('━' * 60)}")
    print(cyan("Demo 3: Counterfactual (Smoking & Cancer)"))
    print(cyan("━" * 60))
    print(agent.load_scenario("smoking"))
    # Build SCM manually
    scm = linear_scm(
        agent.dag,
        coefficients={
            "S": {"G": 0.3},
            "T": {"S": 0.8},
            "C": {"T": 0.5, "S": 0.2, "G": 0.4},
        },
        noise_std=0.1,
    )
    agent.scm = scm
    print(f"\n{agent.ask_effect('S', 'C')}")
    print(f"\n{agent.what_if({'S': 0.0}, 'C')}")
    cf_result = agent.what_if(
        {'S': 0.0}, 'C',
        observed={'G': 1.0, 'S': 2.0, 'T': 1.8, 'C': 3.0},
    )
    print(f"\n{cf_result}")


# ── interactive mode ────────────────────────────────────────────

def run_interactive(initial_query: Optional[str] = None):
    agent = CausalAgent()
    print(bold("=" * 60))
    print(bold("  Causal Inference Agent"))
    print(bold("  Type 'help' for commands, 'quit' to exit"))
    print(bold("=" * 60))

    if initial_query:
        print(f"\n{cyan('>')} {initial_query}")
        print(_process(agent, initial_query))

    while True:
        try:
            user_input = input(f"\n{cyan('>')} ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            print(_process(agent, user_input))
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(red(f"Error: {e}"))


def _process(agent: CausalAgent, user_input: str) -> str:
    cmd = user_input.lower().split()[0] if user_input.split() else ""
    rest = user_input[len(cmd):].strip()

    if cmd in ("help", "h"):
        return (
            f"{bold('Commands:')}\n"
            f"  load <description>   — describe a causal scenario\n"
            f"  template <name>      — load a pre-built template\n"
            f"  load_data <file.csv> — load data for estimation\n"
            f"  discover <file.csv>  — learn causal structure (pc|fci|ges)\n"
            f"  discover ... --bootstrap=50 — with edge confidence\n"
            f"  effect <X> <Y>       — identify + estimate effect\n"
            f"  whatif <X=val> <Y>   — predict Y under intervention\n"
            f"  whatif <X=val> <Y> given <obs> — counterfactual\n"
            f"  explain [concept]    — explain a causal concept\n"
            f"  model                — show current DAG\n"
            f"  dag show             — ASCII art of DAG\n"
            f"  dag save [png] [path]— export DAG to PNG/SVG\n"
            f"  demo                 — run demonstrations\n"
            f"  quit                 — exit\n"
            f"\n{yellow('Templates:')} smoking, simpson, education, frontdoor, mbias"
        )

    if cmd in ("load", "scenario"):
        return agent.load_scenario(rest)

    if cmd in ("template", "t"):
        return agent.load_scenario(rest)

    if cmd in ("load_data", "data", "ld"):
        return agent.load_data(rest)

    if cmd in ("effect", "e", "ask"):
        parts = rest.split()
        if len(parts) >= 2:
            method = "auto"
            if len(parts) >= 3 and parts[2] in ("linear","psm","ipw","dr","stratified","dml","cate_s","cate_t","cate_x","cate_forest"):
                method = parts[2]
                parts = parts[:2]
            return agent.ask_effect(parts[0], parts[1],
                                    data=agent.data, method=method)
        return "Usage: effect <treatment> <outcome> [method]"

    if cmd in ("whatif", "what-if", "wi", "w"):
        # Parse: whatif X=1.0 Y  or  whatif X=1.0 Y given X=2.0,Y=3.0
        parts = rest.split()
        intervention: Dict[str, float] = {}
        observed: Optional[Dict[str, float]] = None
        target = ""
        mode = "intv"
        i = 0
        while i < len(parts):
            if "=" in parts[i] and mode in ("intv", "given"):
                var, val = parts[i].split("=", 1)
                try:
                    if mode == "intv":
                        intervention[var] = float(val)
                    else:
                        if observed is None:
                            observed = {}
                        observed[var] = float(val)
                except ValueError:
                    return f"Invalid value: {val}"
            elif parts[i].lower() == "given":
                mode = "given"
            else:
                if target:
                    return f"Unexpected token: {parts[i]}"
                target = parts[i]
            i += 1

        if not target:
            return "Usage: whatif <X=val> <target> [given <obs>]"
        return agent.what_if(intervention, target, observed)

    if cmd in ("explain", "x"):
        return agent.explain(rest)

    if cmd in ("discover", "disc", "pc", "ges", "fci"):
        parts = rest.split()
        method = "pc"
        bootstrap = 0
        filepath = rest
        remaining_parts = list(parts)
        # Parse method
        if remaining_parts and remaining_parts[0] in ("pc", "fci", "ges"):
            method = remaining_parts[0]
            remaining_parts = remaining_parts[1:]
        # Parse bootstrap
        if remaining_parts and remaining_parts[0].startswith("--bootstrap="):
            try:
                bootstrap = int(remaining_parts[0].split("=")[1])
            except ValueError:
                pass
            remaining_parts = remaining_parts[1:]
        filepath = " ".join(remaining_parts)
        if not filepath:
            return "Usage: discover <file.csv> [pc|fci|ges] [--bootstrap=N]"
        return agent.discover(filepath, method, bootstrap=bootstrap)

    if cmd in ("model", "dag", "m"):
        if agent.dag:
            return agent.dag.summary()

    if cmd in ("dag_show", "dag show", "show", "ascii"):
        if agent.dag:
            return dag_to_ascii(agent.dag)
        return "No model loaded."

    if cmd in ("dag_save", "dag save", "render"):
        if not agent.dag:
            return "No model loaded."
        parts = rest.split()
        fmt = parts[0] if parts and parts[0] in ("png","svg","pdf","dot") else "png"
        fname = parts[1] if len(parts) > 1 else f"/tmp/dag.{fmt}"
        path = render_dag(agent.dag, fname, fmt=fmt)
        if path:
            return green(f"DAG saved to {path}")
        return yellow("graphviz not available; install with: pip install graphviz")

    if cmd in ("demo", "d"):
        run_demo()
        return ""

    # Try parsing as a direct query
    if "causes" in user_input.lower() or "affects" in user_input.lower():
        return agent.load_scenario(user_input)

    return f"Unknown command: {cmd}. Type 'help' for available commands."


# ── main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--demo" in sys.argv or "-d" in sys.argv:
        run_demo()
    elif len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        run_interactive(query)
    else:
        run_interactive()

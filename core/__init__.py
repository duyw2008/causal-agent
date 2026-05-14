"""Causal Agent — core causal inference modules."""
from .graph import CausalDAG
from .scm import SCM, linear_scm, StructuralEquation
from .identification import identify_effect

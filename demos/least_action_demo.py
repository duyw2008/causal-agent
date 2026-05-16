#!/usr/bin/env python3
"""Demo: Principle of Least Action — Pendulum & Harmonic Oscillator.

Validates that the true physical trajectory minimizes the action functional
S = ∫ L dt, and demonstrates how non-physical paths are detected.

Usage: python demos/least_action_demo.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.physics import (
    LagrangianSystem, ActionPrinciple,
    simple_pendulum, harmonic_oscillator,
    PhysicsLibrary,
)


def demo_pendulum():
    """Simple pendulum: θ̈ + (g/l)sinθ = 0.  Compare physical vs non-physical paths."""
    print("=" * 60)
    print("  PRINCIPLE OF LEAST ACTION — Simple Pendulum")
    print("=" * 60)

    l, g = 1.0, 9.81
    omega = np.sqrt(g / l)
    system = simple_pendulum(l=l, g=g)
    principle = ActionPrinciple(system, tolerance=0.02)

    # ── Generate paths ──
    theta_0 = 0.5
    n_steps = 100
    T_final = 0.95    # ωT ≈ 2.98 rad — just under half-period (still a minimum)
    dt = T_final / (n_steps - 1)
    t = np.linspace(0, T_final, n_steps)

    path_physical = theta_0 * np.cos(omega * t)
    path_linear = np.linspace(theta_0, path_physical[-1], n_steps)
    rng = np.random.default_rng(42)
    path_noisy = path_physical + rng.normal(0, 0.05, n_steps)

    print(f"\n  {'Path':<18} {'Action':>12} {'max|δS/δq|':>14} {'Valid':>8}")
    print("  " + "-" * 52)

    results = {}
    for name, path in [("A: Physical", path_physical),
                        ("B: Linear interp", path_linear),
                        ("C: Physical+noise", path_noisy)]:
        S = principle.compute_action(path, dt)
        result = principle.validate_trajectory(path, dt)
        status = "✓" if result["valid"] else "✗"
        print(f"  {name:<18} {S:>12.6f} {result['max_gradient']:>14.2e} {status:>8}")
        results[name] = result

    # ── Physical path should have LOWER action (δS=0, local minimum) ──
    S_phys = results["A: Physical"]["action"]
    S_lin = results["B: Linear interp"]["action"]
    S_noise = results["C: Physical+noise"]["action"]
    g_phys = results["A: Physical"]["max_gradient"]
    g_lin = results["B: Linear interp"]["max_gradient"]

    print(f"\n  ── Verification ──")
    print(f"  S_physical = {S_phys:.6f}  (lowest)")
    print(f"  S_linear   = {S_lin:.6f}  (ΔS={S_lin-S_phys:.6f})")
    print(f"  S_noisy    = {S_noise:.2f}")
    print(f"  max|δS/δq| physical = {g_phys:.2e}  "
          f"({'<<' if g_phys < g_lin/10 else '<'} linear = {g_lin:.2e})")

    assert results["A: Physical"]["valid"], "Physical path must be valid!"
    assert S_phys < S_lin, f"Physical path must minimize action (S_phys={S_phys:.4f} vs S_lin={S_lin:.4f})"
    assert not results["C: Physical+noise"]["valid"], "Noisy path must be invalid!"
    print("  ✓ All assertions passed")


def demo_harmonic():
    """Harmonic oscillator: ẍ + ω²x = 0.  x(t) = cos(ωt)."""
    print(f"\n{'='*60}")
    print("  PRINCIPLE OF LEAST ACTION — Harmonic Oscillator")
    print("=" * 60)

    m, k = 1.0, 4.0
    omega = np.sqrt(k / m)
    system = harmonic_oscillator(m=m, k=k)
    principle = ActionPrinciple(system, tolerance=0.01)

    n_steps = 200
    T_final = 0.3     # ≈ 0.1 period → physical path ≈ linear (action minimum)
    dt = T_final / (n_steps - 1)
    t = np.linspace(0, T_final, n_steps)

    path_physical = np.cos(omega * t)
    path_linear = np.linspace(1.0, np.cos(omega * T_final), n_steps)

    S_phys = principle.compute_action(path_physical, dt)
    S_lin = principle.compute_action(path_linear, dt)

    print(f"\n  Physical action: {S_phys:.6f}")
    print(f"  Linear action:   {S_lin:.6f}")
    print(f"  ΔS:              {S_phys - S_lin:.6f}")
    print(f"  Physical path has lower action: {S_phys < S_lin}")

    result = principle.validate_trajectory(path_physical, dt)
    print(f"  Physical path valid: {result['valid']}  (max|δS/δq|={result['max_gradient']:.2e})")

    assert result["valid"], "Harmonic oscillator physical path must be valid!"
    assert S_phys < S_lin, (
        f"Physical path must have lower action at T={T_final}s "
        f"(S_phys={S_phys:.4f}, S_lin={S_lin:.4f})")
    print("  ✓ All assertions passed")


def demo_physics_library():
    """Verify that least_action law is registered."""
    print(f"\n{'='*60}")
    print("  PhysicsLibrary — Variational Law Check")
    print("=" * 60)

    lib = PhysicsLibrary()
    law = lib.get("least_action")
    assert law is not None, "least_action law not found!"
    assert law.constraint_type.value == "variational"
    print(f"  Law: {law.name}")
    print(f"  Domain: {law.domain}")
    print(f"  LaTeX: {law.latex}")
    print(f"  Type: {law.constraint_type}")
    print(f"  Library count: {len(lib.list_all())} laws")
    print("  ✓ least_action law registered")


if __name__ == "__main__":
    demo_pendulum()
    demo_harmonic()
    demo_physics_library()
    print(f"\n{'='*60}")
    print("  ALL DEMOS PASSED")
    print("=" * 60)

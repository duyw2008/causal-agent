"""
Sensitivity Analysis — how robust are causal conclusions to unobserved confounding?

Methods:
  - Rosenbaum Bounds (binary treatment, matched pairs)
  - E-value (VanderWeele & Ding, 2017)
  - Partial R² bounds (Cinelli & Hazlett, 2020) — simplified
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class SensitivityResult:
    """Result of a sensitivity analysis."""

    method: str
    conclusion: str  # human-readable summary
    details: str     # technical details

    # Rosenbaum
    gamma_threshold: Optional[float] = None  # Γ where p > 0.05

    # E-value
    e_value: Optional[float] = None          # minimum strength of confounding to explain away
    e_value_ci: Optional[float] = None       # E-value for the confidence interval


def rosenbaum_bounds(
    ate: float,
    se: float,
    n_pairs: Optional[int] = None,
    gamma_max: float = 5.0,
    n_gamma: int = 50,
) -> SensitivityResult:
    """
    Rosenbaum bounds for matched-pair studies with binary treatment.

    For each Γ (gamma), computes the upper bound on the p-value
    under the assumption that hidden bias multiplies odds of
    treatment by at most Γ.

    Γ = 1 → no hidden bias (random assignment)
    Γ = 2 → treated unit could be 2× more likely to receive treatment
             due to unobserved confounders

    Returns the Γ where the upper-bound p-value crosses 0.05.
    """
    # Standardized deviate
    if se < 1e-12:
        deviate = abs(ate) / 1e-12
    else:
        deviate = abs(ate) / se

    gamma_values = np.linspace(1.0, gamma_max, n_gamma)
    crossing_gamma = gamma_max

    for gamma in gamma_values:
        try:
            from scipy import stats
        except ImportError:
            # Fallback: rough approximation without scipy
            p_upper = 1.0 / (1.0 + gamma)
            if p_upper < 0.05:
                break
            continue
        # Upper bound on the test statistic under hidden bias Γ
        # For signed rank test (simplified normal approximation)
        p_plus = gamma / (1 + gamma)  # max probability under Γ
        p_minus = 1 / (1 + gamma)     # min probability under Γ

        # Expected value and variance under Γ
        # (simplified: normal approximation for large samples)
        # Upper bound deviate
        if n_pairs:
            mu = n_pairs * (p_plus - p_minus)
            sigma = np.sqrt(n_pairs * p_plus * p_minus * 4)
            upper_deviate = (deviate * se * np.sqrt(n_pairs) - mu) / max(sigma, 1e-12)
        else:
            # Without pair count, use a rough approximation
            # Γ scales the effective sample size
            effective_deviate = deviate / np.sqrt(gamma)
            upper_deviate = effective_deviate

        p_upper = 1 - stats.norm.cdf(abs(upper_deviate))

        if p_upper > 0.05:
            crossing_gamma = gamma
            break

    if crossing_gamma >= gamma_max:
        conclusion = (
            f"Results are robust to hidden bias up to Γ = {gamma_max}. "
            f"Even strong unobserved confounding cannot explain away the effect."
        )
        details = (
            f"At Γ = {gamma_max}, the upper-bound p-value is still ≤ 0.05.\n"
            f"This means: an unobserved confounder would need to make treatment "
            f"more than {gamma_max}× more likely to overturn the result."
        )
    elif crossing_gamma <= 1.1:
        conclusion = (
            f"Results are HIGHLY SENSITIVE to hidden bias. "
            f"Even slight unobserved confounding (Γ ≈ {crossing_gamma:.1f}) "
            f"could explain away the effect."
        )
        details = (
            f"At Γ = {crossing_gamma:.1f}, the result loses statistical significance.\n"
            f"This means: if an unobserved confounder makes treatment just "
            f"{crossing_gamma:.1f}× more likely, the apparent effect could be spurious."
        )
    else:
        conclusion = (
            f"Results are moderately robust. Hidden bias of Γ ≈ {crossing_gamma:.1f} "
            f"would be needed to explain away the effect."
        )
        details = (
            f"The upper-bound p-value crosses 0.05 at Γ = {crossing_gamma:.1f}.\n"
            f"An unobserved confounder would need to change treatment odds by "
            f"a factor of {crossing_gamma:.1f} to nullify the result."
        )

    return SensitivityResult(
        method="Rosenbaum Bounds",
        conclusion=conclusion,
        details=details,
        gamma_threshold=float(crossing_gamma),
    )


def e_value(
    ate: float,
    se: float,
    scale: str = "risk_ratio",
) -> SensitivityResult:
    """
    Compute the E-value (VanderWeele & Ding, 2017).

    The E-value answers:
    "How strong would an unmeasured confounder need to be
     (in terms of its association with BOTH treatment and outcome)
     to explain away the observed effect?"

    E-value = RR + sqrt(RR * (RR - 1))
    where RR = exp(ATE) for risk ratio scale.

    Rule of thumb:
      E-value < 1.5  →  fragile
      E-value 1.5-3  →  moderately robust
      E-value > 3    →  robust
    """
    if se < 1e-12:
        se = 1e-12

    # ATE on risk ratio scale
    if scale == "risk_ratio":
        rr = np.exp(ate)
        rr_ci_lower = np.exp(ate - 1.96 * se)
    else:
        rr = abs(ate)
        rr_ci_lower = abs(ate) - 1.96 * se

    # E-value for point estimate
    if rr > 1:
        e_val = rr + np.sqrt(rr * (rr - 1))
    else:
        rr_inv = 1 / max(rr, 1e-6)
        e_val = rr_inv + np.sqrt(rr_inv * (rr_inv - 1))

    # E-value for CI bound
    if rr_ci_lower > 1:
        e_val_ci = rr_ci_lower + np.sqrt(rr_ci_lower * (rr_ci_lower - 1))
    elif 1 / max(rr_ci_lower, 1e-6) > 1:
        e_val_ci = (1 / rr_ci_lower) + np.sqrt((1 / rr_ci_lower) * ((1 / rr_ci_lower) - 1))
    else:
        e_val_ci = 1.0

    # Interpret
    if e_val > 5:
        conclusion = (
            f"Highly robust (E-value = {e_val:.1f}). "
            f"An unmeasured confounder would need to be associated with BOTH "
            f"treatment and outcome by a risk ratio of >{e_val:.1f} each "
            f"to explain away the observed effect."
        )
    elif e_val > 2:
        conclusion = (
            f"Moderately robust (E-value = {e_val:.1f}). "
            f"Moderate unobserved confounding could potentially explain the result."
        )
    else:
        conclusion = (
            f"Fragile (E-value = {e_val:.1f}). "
            f"Even weak unobserved confounding could explain away the effect."
        )

    details = (
        f"E-value (point estimate): {e_val:.2f}\n"
        f"E-value (CI limit): {e_val_ci:.2f}\n"
        f"Interpretation: To reduce the observed ATE to zero, an unmeasured\n"
        f"confounder must be associated with BOTH treatment and outcome by\n"
        f"risk ratios of at least {e_val:.1f} (above and beyond measured covariates).\n"
        f"To shift the CI to include zero: at least {e_val_ci:.1f}."
    )

    return SensitivityResult(
        method="E-value",
        conclusion=conclusion,
        details=details,
        e_value=float(e_val),
        e_value_ci=float(e_val_ci),
    )


def full_sensitivity_report(
    ate: float,
    se: float,
    n: Optional[int] = None,
) -> str:
    """Generate a complete sensitivity analysis report."""
    lines = [
        "=" * 55,
        "  SENSITIVITY ANALYSIS",
        "=" * 55,
        f"  ATE = {ate:.4f}  (SE = {se:.4f})",
        "",
    ]

    # Rosenbaum
    rb = rosenbaum_bounds(ate, se)
    lines.append(f"── Rosenbaum Bounds ──")
    lines.append(f"  {rb.conclusion}")
    lines.append(f"  Γ threshold: {rb.gamma_threshold:.2f}")
    lines.append("")

    # E-value
    ev = e_value(ate, se)
    lines.append(f"── E-value ──")
    lines.append(f"  {ev.conclusion}")
    lines.append(f"  E-value: {ev.e_value:.2f}")
    lines.append("")

    lines.append("=" * 55)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  SENSITIVITY ANALYSIS TESTS")
    print("=" * 55)

    # Strong effect
    print("\n── Strong effect (ATE=2.0, SE=0.2) ──")
    print(full_sensitivity_report(2.0, 0.2))

    # Weak effect
    print("\n── Weak effect (ATE=0.3, SE=0.15) ──")
    print(full_sensitivity_report(0.3, 0.15))

    # Moderate
    print("\n── Moderate (ATE=0.8, SE=0.3) ──")
    print(full_sensitivity_report(0.8, 0.3))

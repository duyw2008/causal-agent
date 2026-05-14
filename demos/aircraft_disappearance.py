"""
Aircraft Disappearance — Time Series Event Prediction

Scenario:
  Ground radar tracks an aircraft. The aircraft enters a cloud bank
  and disappears from radar. We observe the trajectory BEFORE cloud entry.
  Task: predict WHY it disappeared and estimate probability of each cause.

Causal Structure (ground truth, for data generation):

                    ┌──────────────┐
                    │   Weather    │  (cloud density, storm)
                    │   Severity   │
                    └──┬───────┬───┘
                       │       │
              ┌────────▼──┐    │    ┌──────────────┐
              │ Visibility │    │    │  Mechanical  │ (engine health)
              │  (radar)   │    │    │   Failure    │
              └────────────┘    │    └──────┬───────┘
                                │           │
              ┌─────────────────▼───────────▼──────────┐
              │           Trajectory Features          │
              │  altitude(t), speed(t), heading(t),    │
              │  vertical_accel(t), deviation(t)       │
              └────────────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     Disappearance Cause     │
                    │  normal / crash / divert /  │
                    │  mechanical / weather_loss  │
                    └─────────────────────────────┘

Key insight:
  - Weather → Visibility (direct)
  - Weather → Trajectory (turbulence, pilot reaction)
  - Mechanical Failure → Trajectory (loss of control)
  - Trajectory features → Disappearance cause
  - We only observe Trajectory BEFORE cloud entry → censored data
  - Causal discovery from data should recover Weather→Traj and Mech→Traj edges

Five disappearance causes with distinct trajectory signatures:

  Cause          | Altitude | Speed  | Heading | Vert Accel | Probability
  ───────────────┼──────────┼────────┼─────────┼────────────┼───────────
  normal         | stable   | stable | stable  | ~0         | 60%
  crash          | ↓↓↓      | ↓↓     | wobble  | ↓↓↓        | 15%
  divert         | stable   | stable | turn →  | ~0         | 10%
  mechanical     | ↓↓       | ↓↓↓    | erratic | ↓↓         | 10%
  weather_loss   | ±bumpy   | ↓      | wobble  | ±bumpy     | 5%
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════
#  Data Generation
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TrajectoryConfig:
    """Configuration for trajectory data generation."""
    n_samples: int = 5000          # Number of flights
    track_length: int = 120        # Radar observations before cloud entry
    dt: float = 1.0               # Time step (seconds)

    # Base flight parameters
    cruise_altitude: float = 10000.0   # ft
    cruise_speed: float = 450.0        # knots
    cruise_heading: float = 90.0       # degrees (east)

    # Weather severity distribution
    weather_mean: float = 0.3
    weather_std: float = 0.2

    # Mechanical failure probability
    mech_failure_prob: float = 0.08

    # Cause probabilities (before weather/mech effects)
    base_cause_probs: Dict[str, float] = field(default_factory=lambda: {
        "normal": 0.60,
        "crash": 0.15,
        "divert": 0.10,
        "mechanical": 0.10,
        "weather_loss": 0.05,
    })

    # Noise levels
    altitude_noise: float = 50.0       # ft
    speed_noise: float = 5.0           # knots
    heading_noise: float = 1.0         # degrees


def generate_trajectory_dataset(
    config: Optional[TrajectoryConfig] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Generate synthetic aircraft trajectory dataset.

    Returns
    -------
    trajectories : (n_samples, track_length, n_features)
        Time series of: [altitude, speed, heading, vertical_accel, deviation]
    causes : (n_samples,) int
        Cause label: 0=normal, 1=crash, 2=divert, 3=mechanical, 4=weather_loss
    metadata : (n_samples, n_meta)
        [weather_severity, mech_failure, visibility]
    info : dict
        Cause mapping and config details.
    """
    if config is None:
        config = TrajectoryConfig()
    cfg = config
    rng = np.random.default_rng(seed)

    n = cfg.n_samples
    T = cfg.track_length

    # ── Step 1: Generate exogenous variables ─────────────────
    # Weather severity ∈ [0,1]
    weather = np.clip(
        rng.normal(cfg.weather_mean, cfg.weather_std, n), 0.0, 1.0
    )
    # Mechanical failure (binary, with some correlation to weather)
    mech_base = rng.random(n)
    mech_failure = (mech_base < cfg.mech_failure_prob).astype(float)

    # Visibility: determined by weather
    visibility = np.clip(1.0 - weather + rng.normal(0, 0.1, n), 0.0, 1.0)

    # ── Step 2: Sample cause based on weather and mech ────────
    # Weather ↑ → more likely weather_loss
    # Mech failure → mechanical cause
    causes = np.zeros(n, dtype=int)
    cause_names = ["normal", "crash", "divert", "mechanical", "weather_loss"]

    for i in range(n):
        probs = dict(cfg.base_cause_probs)
        # Adjust probs based on exogenous
        if mech_failure[i] > 0.5:
            probs["mechanical"] = min(1.0, probs["mechanical"] + 0.6)
            probs["normal"] = max(0.0, probs["normal"] - 0.4)
        if weather[i] > 0.5:
            probs["weather_loss"] = min(1.0, probs["weather_loss"] + 0.3)
            probs["normal"] = max(0.0, probs["normal"] - 0.15)
            probs["crash"] = min(1.0, probs["crash"] + 0.05)

        # Normalize
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}
        keys = list(probs.keys())
        vals = np.array([probs[k] for k in keys])
        causes[i] = keys.index(
            rng.choice(keys, p=vals / vals.sum())
        )

    # ── Step 3: Generate trajectories ─────────────────────────
    trajectories = np.zeros((n, T, 5))  # alt, speed, heading, v_accel, dev

    # Base trajectory: straight and level
    t = np.arange(T) * cfg.dt

    for i in range(n):
        cause = causes[i]
        w = weather[i]
        m = mech_failure[i]

        alt = np.full(T, cfg.cruise_altitude)
        spd = np.full(T, cfg.cruise_speed)
        hdg = np.full(T, cfg.cruise_heading)

        # Apply cause-specific trajectory patterns
        # (effects become more pronounced closer to cloud entry)

        if cause == 0:  # normal
            # Minor weather-induced turbulence
            alt += rng.normal(0, cfg.altitude_noise * (1 + w), T)
            spd += rng.normal(0, cfg.speed_noise * (1 + w), T)
            hdg += rng.normal(0, cfg.heading_noise, T)

        elif cause == 1:  # crash — dramatic altitude loss toward end
            severity = rng.uniform(0.3, 0.8)
            ramp = np.linspace(0, 1, T) ** 2
            alt_loss = ramp * severity * 3000
            alt -= alt_loss
            spd -= ramp * severity * 100
            hdg += np.cumsum(rng.normal(0, 2.0, T))
            alt += rng.normal(0, cfg.altitude_noise, T)
            spd += rng.normal(0, cfg.speed_noise, T)

        elif cause == 2:  # divert — heading change
            turn_start = int(T * rng.uniform(0.3, 0.7))
            turn_rate = rng.uniform(1.5, 4.0) * rng.choice([-1, 1])
            for j in range(turn_start, T):
                hdg[j:] += turn_rate * cfg.dt
            alt += rng.normal(0, cfg.altitude_noise, T)
            spd += rng.normal(0, cfg.speed_noise, T)

        elif cause == 3:  # mechanical — speed loss + erratic altitude
            failure_start = int(T * rng.uniform(0.2, 0.5))
            ramp = np.zeros(T)
            ramp[failure_start:] = np.linspace(0, 1, T - failure_start) ** 1.5
            spd -= ramp * 150
            alt -= ramp * 1500
            hdg += np.cumsum(rng.normal(0, 3.0, T)) * ramp
            alt += rng.normal(0, cfg.altitude_noise * 2, T)

        elif cause == 4:  # weather_loss — severe turbulence
            turbulence = w * rng.normal(0, 1, T) * 300
            alt += np.cumsum(turbulence) * 0.3
            spd -= w * 50 + rng.normal(0, cfg.speed_noise * 2, T)
            hdg += np.cumsum(rng.normal(0, 1.5, T))
            alt += rng.normal(0, cfg.altitude_noise, T)

        # Vertical acceleration: derivative of altitude rate
        alt_rate = np.gradient(alt, cfg.dt)
        v_accel = np.gradient(alt_rate, cfg.dt)

        # Deviation from straight path
        expected_hdg = cfg.cruise_heading
        deviation = np.abs(hdg - expected_hdg)
        # wrap around 360
        deviation = np.minimum(deviation, 360 - deviation)

        trajectories[i, :, 0] = alt / 1000.0       # kft
        trajectories[i, :, 1] = spd / 100.0        # 100 kts
        trajectories[i, :, 2] = hdg / 10.0         # 10 deg
        trajectories[i, :, 3] = v_accel / 100.0    # 100 ft/s²
        trajectories[i, :, 4] = deviation / 10.0   # 10 deg

    # Metadata
    metadata = np.column_stack([weather, mech_failure, visibility])

    info = {
        "cause_names": cause_names,
        "feature_names": ["altitude", "speed", "heading", "vert_accel", "deviation"],
        "feature_units": ["kft", "100kts", "10deg", "100ft/s²", "10deg"],
        "config": cfg,
    }

    return trajectories, causes, metadata, info


# ═══════════════════════════════════════════════════════════════════
#  Feature Extraction for Time Series
# ═══════════════════════════════════════════════════════════════════

def extract_trajectory_features(
    trajectories: np.ndarray,
) -> np.ndarray:
    """
    Extract hand-crafted features from trajectory time series.

    For each trajectory, compute:
      - Last N values (most recent observations)
      - Trends (linear slope over last K points)
      - Statistics (mean, std, min, max, range over window)
      - Rate of change at end

    Returns (n_samples, n_features) feature matrix.
    """
    n, T, D = trajectories.shape

    features_list = []

    for i in range(n):
        traj = trajectories[i]  # (T, D)
        feats = []

        for d in range(D):
            series = traj[:, d]

            # Last value
            feats.append(series[-1])

            # Trend over last 30, 60 points
            for window in [30, 60]:
                if T >= window:
                    w = series[-window:]
                    x = np.arange(window)
                    slope = np.polyfit(x, w, 1)[0]
                    feats.append(slope)
                else:
                    feats.append(0.0)

            # Statistics over last 60 points
            if T >= 60:
                w = series[-60:]
                feats.extend([np.mean(w), np.std(w), np.min(w), np.max(w)])
            else:
                feats.extend([0.0, 0.0, 0.0, 0.0])

            # Rate of change (last 10 points)
            if T >= 10:
                roc = np.mean(np.diff(series[-10:]))
                feats.append(roc)
            else:
                feats.append(0.0)

            # Maximum deviation from mean (last 30)
            if T >= 30:
                w = series[-30:]
                feats.append(np.max(np.abs(w - np.mean(w))))
            else:
                feats.append(0.0)

        features_list.append(feats)

    return np.array(features_list)


# ═══════════════════════════════════════════════════════════════════
#  Event Probability Model
# ═══════════════════════════════════════════════════════════════════

class TrajectoryEventPredictor:
    """
    Predict probability of each disappearance cause given trajectory.

    Uses a simple multinomial logistic regression (softmax).
    """

    def __init__(self, n_classes: int = 5):
        self.n_classes = n_classes
        self.weights: Optional[np.ndarray] = None
        self.bias: Optional[np.ndarray] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.01,
        epochs: int = 500,
        lambda_reg: float = 0.01,
        verbose: bool = False,
    ):
        """
        Train multinomial logistic regression via gradient descent.

        Parameters
        ----------
        X : (n_samples, n_features) — trajectory features
        y : (n_samples,) — cause labels (0..4)
        """
        n, d = X.shape

        # Standardize
        self.feature_mean = X.mean(axis=0)
        self.feature_std = X.std(axis=0) + 1e-8
        X_norm = (X - self.feature_mean) / self.feature_std

        # One-hot encode y
        Y = np.zeros((n, self.n_classes))
        Y[np.arange(n), y.astype(int)] = 1

        # Initialize
        rng = np.random.default_rng(42)
        self.weights = rng.normal(0, 0.01, (d, self.n_classes))
        self.bias = np.zeros(self.n_classes)

        losses = []
        for epoch in range(epochs):
            # Forward
            logits = X_norm @ self.weights + self.bias
            logits_max = logits.max(axis=1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            # Cross-entropy loss + L2
            loss = -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))
            loss += lambda_reg * np.sum(self.weights ** 2)
            losses.append(loss)

            # Gradient
            d_logits = (probs - Y) / n
            dW = X_norm.T @ d_logits + 2 * lambda_reg * self.weights
            db = d_logits.sum(axis=0)

            self.weights -= lr * dW
            self.bias -= lr * db

            if verbose and epoch % 100 == 0:
                acc = (probs.argmax(axis=1) == y).mean()
                print(f"  Epoch {epoch:4d}: loss={loss:.4f}, acc={acc:.3f}")

        self.train_losses = losses
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities. Returns (n_samples, n_classes)."""
        X_norm = (X - self.feature_mean) / self.feature_std
        logits = X_norm @ self.weights + self.bias
        logits_max = logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict most likely cause."""
        return self.predict_proba(X).argmax(axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy."""
        return (self.predict(X) == y).mean()

    def feature_importance(self) -> np.ndarray:
        """Average absolute weight per class × input std."""
        return np.abs(self.weights).mean(axis=1)


# ═══════════════════════════════════════════════════════════════════
#  Causal Analysis
# ═══════════════════════════════════════════════════════════════════

def causal_analysis_report(
    trajectories: np.ndarray,
    causes: np.ndarray,
    metadata: np.ndarray,
    info: Dict,
) -> str:
    """Generate a causal analysis report."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.graph import CausalDAG
    from core.identification import identify_effect

    lines = []
    lines.append("=" * 60)
    lines.append("  CAUSAL ANALYSIS — Aircraft Disappearance")
    lines.append("=" * 60)

    # ── True causal graph ──
    dag = CausalDAG(
        ["Weather", "MechFail", "Visibility",
         "AltTrend", "SpeedTrend", "HeadingTrend", "VAccel", "Deviation",
         "Cause"],
        [
            ("Weather", "Visibility"),
            ("Weather", "AltTrend"),
            ("Weather", "SpeedTrend"),
            ("Weather", "HeadingTrend"),
            ("Weather", "VAccel"),
            ("MechFail", "AltTrend"),
            ("MechFail", "SpeedTrend"),
            ("MechFail", "HeadingTrend"),
            ("MechFail", "VAccel"),
            ("AltTrend", "Cause"),
            ("SpeedTrend", "Cause"),
            ("HeadingTrend", "Cause"),
            ("VAccel", "Cause"),
            ("Deviation", "Cause"),
        ],
    )
    lines.append("\nTrue Causal DAG:")
    lines.append(dag.summary())

    # ── Identify effects ──
    lines.append("\n── Causal Effect Identification ──")

    for treatment in ["Weather", "MechFail"]:
        result = identify_effect(dag, treatment, "Cause")
        lines.append(f"\n  Effect of {treatment} on Cause:")
        lines.append(f"    Identifiable: {result.identifiable}")
        if result.adjustment_set:
            lines.append(f"    Adjustment set: {result.adjustment_set}")

    # ── Observational vs interventional ──
    lines.append("\n── Observational Probabilities ──")
    cause_names = info["cause_names"]
    for k in range(5):
        prob = (causes == k).mean()
        lines.append(f"  P(Cause={cause_names[k]}) = {prob:.3f}")

    # Weather-stratified
    lines.append("\n── Weather-stratified Probabilities ──")
    w = metadata[:, 0]
    for weather_level, label in [(0.3, "Low weather"), (0.7, "High weather")]:
        mask = w > weather_level
        if mask.sum() > 0:
            lines.append(f"\n  {label} (n={mask.sum()}):")
            for k in range(5):
                prob = (causes[mask] == k).mean()
                lines.append(f"    P(Cause={cause_names[k]}) = {prob:.3f}")

    # ── Model performance ──
    lines.append(f"\n  NOTE: Observational P(Cause|Weather) ≠ P(Cause|do(Weather))")
    lines.append(f"        because Weather→Trajectory→Cause AND Weather→...→Cause")
    lines.append(f"        The back-door path through MechFail needs adjustment.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating aircraft trajectory dataset...")
    traj, causes, meta, info = generate_trajectory_dataset(seed=42)

    print(f"  Trajectories: {traj.shape}")
    print(f"  Cause distribution: {dict(zip(info['cause_names'], np.bincount(causes, minlength=5)))}")

    # Extract features
    X = extract_trajectory_features(traj)
    print(f"  Extracted features: {X.shape[1]} features per sample")

    # Split
    n = len(traj)
    idx = np.random.default_rng(42).permutation(n)
    split = int(0.8 * n)
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = causes[idx[:split]], causes[idx[split:]]

    # Train model
    print("\nTraining event probability model...")
    model = TrajectoryEventPredictor(n_classes=5)
    model.fit(X_train, y_train, lr=0.05, epochs=800, lambda_reg=0.001, verbose=True)

    acc = model.score(X_test, y_test)
    print(f"\n  Test accuracy: {acc:.3f}")

    # Per-class accuracy
    y_pred = model.predict(X_test)
    for k in range(5):
        mask = y_test == k
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == k).mean()
            print(f"  {info['cause_names'][k]:15s}: {class_acc:.3f} ({mask.sum()} samples)")

    # Example prediction
    print("\n── Example Prediction ──")
    idx_example = 0
    probs = model.predict_proba(X_test[idx_example:idx_example+1])[0]
    true_cause = info["cause_names"][y_test[idx_example]]
    for k in range(5):
        marker = " ◀ TRUE" if k == y_test[idx_example] else ""
        print(f"  {info['cause_names'][k]:15s}: {probs[k]:.4f}{marker}")

    # Causal analysis
    print("\n" + causal_analysis_report(traj, causes, meta, info))

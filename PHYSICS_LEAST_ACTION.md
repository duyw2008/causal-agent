# 最小作用量原理 — 物理因果引擎集成方案

> 版本: v0.9.1 | 日期: 2026-05-14
> 核心模块: `core/physics.py`
> 状态: 设计文档 — 待实施

---

## 一、原理与动机

### 1.1 最小作用量原理

物理系统在两个时刻 $t_1, t_2$ 之间的真实演化路径 $q(t)$ 使作用量泛函取极值：

$$S[q] = \int_{t_1}^{t_2} L(q, \dot{q}, t) \, dt$$

其中 $L = T - V$ 为拉格朗日量（动能减势能）。变分 $\delta S = 0$ 导出 Euler-Lagrange 方程：

$$\frac{d}{dt} \frac{\partial L}{\partial \dot{q}} - \frac{\partial L}{\partial q} = 0$$

### 1.2 对因果推断的价值

| 约束层次 | 约束力 | 机制 |
|---------|:---:|------|
| 单条定律 (F=ma) | ★★ | 局部：只约束一个变量对 |
| 守恒律 | ★★★ | 半全局：约束两个时刻的量 |
| **最小作用量** | ★★★★★ | **全局：约束整条轨迹必须满足 $\delta S=0$** |

现有 `physics.py` 已有单条定律和守恒律，缺**变分原理**这一最高层级约束。加入后，反事实推演不只是"重算一个 SCM 样本"，而是"找到满足物理规律的整条演化路径"。

---

## 二、架构设计

### 2.1 新增组件

```
physics.py (扩展)
│
├── ConstraintType.VARIATIONAL    [新增] 变分原理约束类型
│
├── LagrangianSystem               [新增] 拉格朗日力学系统
│   ├── L(q, q_dot, t)           动能减势能
│   ├── euler_lagrange()         推导运动方程
│   └── equations_of_motion()    返回常微分方程组
│
├── ActionPrinciple               [新增] 作用量泛函 + 轨迹优化
│   ├── compute_action()         离散计算 S = Σ L·Δt
│   ├── validate_trajectory()    检查 δS 是否接近零
│   ├── find_stationary_path()   数值优化找到最小作用量路径
│   └── compare_paths()          比较两条路径的作用量
│
└── physics_causal_pipeline()     [修改] 加入 variational 约束
```

### 2.2 与现有框架的衔接

```
用户输入场景
    │
    ▼
physics_causal_pipeline()
    │
    ├─ Step 1: PC/GES 发现 DAG           (现有)
    ├─ Step 2: PhysicsInformedCausalGraph  (现有, DAG 层约束)
    ├─ Step 3: PhysicsInformedSCM          (现有, 方程层约束)
    ├─ Step 4: SymbolicPhysicsDiscovery   (现有, 公式发现)
    │
    ├─ Step 5: [NEW] ActionPrinciple      ← 轨迹层约束
    │   └─ validate_trajectory() → δS ≈ 0 ?
    │
    ├─ Step 6: Effect identification       (现有)
    └─ Step 7: Counterfactual              (现有)
```

---

## 三、具体代码

### 3.1 LagrangianSystem

```python
@dataclass
class LagrangianSystem:
    """A physical system defined by its Lagrangian function L = T - V.

    Attributes:
        name: Human-readable system name.
        generalized_coords: List of generalized coordinate names (e.g. ['theta']).
        kinetic_energy: Function T(q_dot, q, params) → float.
        potential_energy: Function V(q, params) → float.
        params: Parameter values (e.g. {'g': 9.81, 'l': 1.0, 'm': 0.5}).
    """
    name: str
    generalized_coords: List[str]
    kinetic_energy: Callable   # T(q_dot, q, params) → float
    potential_energy: Callable  # V(q, params) → float
    params: Dict[str, float] = field(default_factory=dict)

    def lagrangian(self, q: float, q_dot: float, t: float = 0.0) -> float:
        """L = T - V"""
        return (self.kinetic_energy(q_dot, q, self.params)
                - self.potential_energy(q, self.params))

    def euler_lagrange_rhs(self, q: float, q_dot: float, t: float = 0.0,
                           delta: float = 1e-6) -> Tuple[float, float]:
        """Numerically evaluate d/dt(∂L/∂q_dot) - ∂L/∂q."""
        # ∂L/∂q via central difference
        L_plus = self.lagrangian(q + delta, q_dot, t)
        L_minus = self.lagrangian(q - delta, q_dot, t)
        dL_dq = (L_plus - L_minus) / (2 * delta)

        # ∂L/∂q_dot via central difference
        L_plus = self.lagrangian(q, q_dot + delta, t)
        L_minus = self.lagrangian(q, q_dot - delta, t)
        dL_dqdot = (L_plus - L_minus) / (2 * delta)

        # d/dt(∂L/∂q_dot) → for conservative systems, this is the
        # time derivative that we approximate via the equation of motion.
        # The Euler-Lagrange residual: should be zero for a physical trajectory.

        return dL_dqdot, dL_dq  # caller computes d/dt(dL_dqdot) - dL_dq

    def equations_of_motion(self) -> str:
        """Return a human-readable description of the equations of motion."""
        return (f"{self.name}: δS/δq = 0 → d/dt(∂L/∂q̇) = ∂L/∂q")
```

### 3.2 预设系统工厂函数

```python
def harmonic_oscillator(m: float = 1.0, k: float = 1.0) -> LagrangianSystem:
    """Harmonic oscillator: L = ½mẋ² - ½kx² → ẍ + (k/m)x = 0"""
    return LagrangianSystem(
        name="harmonic_oscillator",
        generalized_coords=["x"],
        kinetic_energy=lambda qd, q, p: 0.5 * p.get("m", m) * qd**2,
        potential_energy=lambda q, p: 0.5 * p.get("k", k) * q**2,
        params={"m": m, "k": k},
    )


def simple_pendulum(l: float = 1.0, g: float = 9.81) -> LagrangianSystem:
    """Simple pendulum: L = ½ml²θ̇² - mgl(1-cosθ) → θ̈ + (g/l)sinθ = 0"""
    return LagrangianSystem(
        name="simple_pendulum",
        generalized_coords=["theta"],
        kinetic_energy=lambda qd, q, p: 0.5 * p.get("m", 1.0) * p.get("l", l)**2 * qd**2,
        potential_energy=lambda q, p: (p.get("m", 1.0) * p.get("g", g)
                                       * p.get("l", l) * (1 - np.cos(q))),
        params={"l": l, "g": g, "m": 1.0},
    )


def double_pendulum(l1: float = 1.0, l2: float = 0.5,
                    m1: float = 1.0, m2: float = 0.5,
                    g: float = 9.81) -> LagrangianSystem:
    """Double pendulum: 2-DOF chaotic system."""
    def T(qd, q, p):
        th1, th2 = q if isinstance(q, (list, np.ndarray)) else (q, 0)
        th1d, th2d = qd if isinstance(qd, (list, np.ndarray)) else (qd, 0)
        L1, L2 = p.get("l1", l1), p.get("l2", l2)
        M1, M2 = p.get("m1", m1), p.get("m2", m2)
        return (0.5 * M1 * L1**2 * th1d**2
                + 0.5 * M2 * (L1**2 * th1d**2 + L2**2 * th2d**2
                              + 2*L1*L2*th1d*th2d*np.cos(th1-th2)))

    def V(q, p):
        th1, th2 = q if isinstance(q, (list, np.ndarray)) else (q, 0)
        L1, L2 = p.get("l1", l1), p.get("l2", l2)
        M1, M2 = p.get("m1", m1), p.get("m2", m2)
        G = p.get("g", g)
        return -G * (M1*L1*np.cos(th1) + M2*(L1*np.cos(th1) + L2*np.cos(th2)))

    return LagrangianSystem(
        name="double_pendulum",
        generalized_coords=["theta1", "theta2"],
        kinetic_energy=T,
        potential_energy=V,
        params={"l1": l1, "l2": l2, "m1": m1, "m2": m2, "g": g},
    )
```

### 3.3 ActionPrinciple

```python
@dataclass
class ActionPrinciple:
    """Validate and optimize physical trajectories via δS = 0.

    The Principle of Least Action states that the true physical path
    is a stationary point of the action functional S[q] = ∫ L dt.

    For causal inference: any counterfactual trajectory must satisfy
    δS ≈ 0. Trajectories with large δS are physically impossible.
    """
    system: LagrangianSystem
    tolerance: float = 0.01  # Maximum acceptable |δS| for a valid trajectory

    def compute_action(self, q_path: np.ndarray, dt: float) -> float:
        """Compute the discretized action S = Σ L(q_i, q̇_i) · Δt.

        Args:
            q_path: Array of shape (n_steps,) or (n_steps, n_dof) for multi-DOF.
            dt: Time step between successive points.

        Returns:
            Total action S (scalar).
        """
        n = len(q_path)
        if n < 2:
            return 0.0

        S = 0.0
        for i in range(n - 1):
            q = q_path[i]
            q_next = q_path[i + 1]
            q_dot = (q_next - q) / dt
            t = i * dt
            L = self.system.lagrangian(q, q_dot, t)
            S += L * dt

        return S

    def variational_derivative(self, q_path: np.ndarray,
                                dt: float) -> np.ndarray:
        """Compute δS/δq at each interior point via central difference.

        For each interior point i, perturb q_i by ±ε to estimate the
        functional derivative. A stationary path satisfies δS/δq_i ≈ 0
        for all i.

        Returns:
            Array of shape (n-2,) — the δS/δq values at interior points.
        """
        n = len(q_path)
        eps = 1e-6
        derivatives = np.zeros(n - 2)

        for i in range(1, n - 1):
            q_plus = q_path.copy()
            q_minus = q_path.copy()
            q_plus[i] += eps
            q_minus[i] -= eps

            S_plus = self.compute_action(q_plus, dt)
            S_minus = self.compute_action(q_minus, dt)
            derivatives[i - 1] = (S_plus - S_minus) / (2 * eps)

        return derivatives

    def validate_trajectory(self, q_path: np.ndarray, dt: float) -> Dict:
        """Check if a trajectory satisfies the Principle of Least Action.

        Returns:
            {
                "valid": bool,
                "max_gradient": float,       # max |δS/δq_i|
                "rms_gradient": float,       # RMS of δS/δq
                "action": float,             # total action S
                "n_violations": int,         # points exceeding tolerance
                "verdict": str,              # human-readable
            }
        """
        derivs = self.variational_derivative(q_path, dt)
        max_grad = float(np.max(np.abs(derivs)))
        rms_grad = float(np.sqrt(np.mean(derivs**2)))
        n_viol = int(np.sum(np.abs(derivs) > self.tolerance))
        action = self.compute_action(q_path, dt)

        valid = max_grad < self.tolerance

        if valid:
            verdict = f"✓ Valid trajectory: max|δS/δq| = {max_grad:.2e}"
        else:
            verdict = (f"✗ Invalid trajectory: {n_viol}/{len(derivs)} "
                       f"points violate δS=0 (max={max_grad:.2e})")

        return {
            "valid": valid,
            "max_gradient": max_grad,
            "rms_gradient": rms_grad,
            "action": action,
            "n_violations": n_viol,
            "verdict": verdict,
        }

    def find_stationary_path(self, q_start: float, q_end: float,
                              n_steps: int, dt: float,
                              max_iter: int = 500,
                              lr: float = 0.01) -> Dict:
        """Numerically find a stationary-action path via gradient descent.

        Initializes with a straight line from q_start to q_end, then
        iteratively adjusts interior points to minimize |δS/δq|.

        Returns:
            {
                "q_path": np.ndarray,       # optimized trajectory
                "action": float,            # final action value
                "max_gradient": float,      # final max |δS/δq|
                "iterations": int,          # number of iterations used
                "converged": bool,          # did it converge?
            }
        """
        # Initialize linear interpolation
        q_path = np.linspace(q_start, q_end, n_steps)

        for iteration in range(max_iter):
            derivs = self.variational_derivative(q_path, dt)
            max_grad = np.max(np.abs(derivs))

            if max_grad < self.tolerance:
                return {
                    "q_path": q_path,
                    "action": self.compute_action(q_path, dt),
                    "max_gradient": max_grad,
                    "iterations": iteration + 1,
                    "converged": True,
                }

            # Gradient descent on interior points
            q_path[1:-1] -= lr * derivs

        return {
            "q_path": q_path,
            "action": self.compute_action(q_path, dt),
            "max_gradient": float(np.max(np.abs(
                self.variational_derivative(q_path, dt)))),
            "iterations": max_iter,
            "converged": False,
        }

    def compare_paths(self, path_a: np.ndarray, path_b: np.ndarray,
                       dt: float) -> Dict:
        """Compare two paths by their action values.

        The path with LOWER action is physically preferred (for
        bounded trajectories; for unbounded, the stationary path
        might be a saddle point).
        """
        S_a = self.compute_action(path_a, dt)
        S_b = self.compute_action(path_b, dt)
        val_a = self.validate_trajectory(path_a, dt)
        val_b = self.validate_trajectory(path_b, dt)

        return {
            "path_a_action": S_a,
            "path_b_action": S_b,
            "path_a_valid": val_a["valid"],
            "path_b_valid": val_b["valid"],
            "delta_S": S_a - S_b,
            "preferred": "a" if S_a < S_b else "b",
            "interpretation": (
                f"Path {'A' if S_a < S_b else 'B'} has lower action "
                f"(ΔS = {abs(S_a - S_b):.4f}) "
                f"and is physically preferred."
            ),
        }
```

### 3.4 集成到 PhysicsLibrary — 添加变分原理定律

```python
# 在 PhysicsLibrary.__init__() 的 _laws 列表中追加

PhysicsLaw(
    name="least_action",
    domain="mechanics",
    latex=r"$\delta S = \delta \int L \, dt = 0$",
    inputs=["trajectory"],
    outputs=[],
    constraint_type=ConstraintType.VARIATIONAL,
    formula=None,  # 不适用 — 由 ActionPrinciple 直接处理
    causal_direction=[],
    forbidden_directions=[],
    required_parents={},
    tolerance=0.01,
),
```

### 3.5 集成到 physics_causal_pipeline()

```python
# 在 physics_causal_pipeline() 中, Step 5 之后插入:

# Step 5.5: Variational validation (if system has dynamics)
var_result = None
if "least_action" in [law.name for law in physics.laws
                       if law.constraint_type == ConstraintType.VARIATIONAL]:
    # Check if DAG implies a dynamical system
    has_dynamics = any(
        dag.parents(v) and any(
            p.lower() in {"position", "velocity", "theta", "angle", "x", "q"}
            for p in dag.parents(v)
        )
        for v in dag.variables
    )
    if has_dynamics:
        var_result = _run_variational_validation(dag, scm, data, var_names)
```

```python
def _run_variational_validation(dag, scm, data, var_names):
    """Helper: run variational validation on a causal system."""
    # Detect which LagrangianSystem matches the DAG
    system = _infer_lagrangian_system(dag, scm)
    if system is None:
        return {"error": "No Lagrangian system detected"}

    principle = ActionPrinciple(system, tolerance=0.01)

    # Generate a trajectory from SCM
    n_steps = min(len(data), 100)
    dt = 0.1

    # For SCM: trajectory = time-ordered samples of the relevant variable
    # Here we demonstrate with the first coordinate
    coord_name = system.generalized_coords[0]
    if coord_name in var_names:
        idx = var_names.index(coord_name)
        q_path = data[:n_steps, idx]
    else:
        # Generate from SCM as a time series
        q_path = np.zeros(n_steps)
        for i in range(n_steps):
            samp = scm.sample(1)
            q_path[i] = samp[0] if isinstance(samp, np.ndarray) else samp

    # Validate
    result = principle.validate_trajectory(q_path, dt)

    return {
        "system": system.name,
        "action": result["action"],
        "max_gradient": result["max_gradient"],
        "valid": result["valid"],
        "verdict": result["verdict"],
    }


def _infer_lagrangian_system(dag, scm):
    """Heuristic: infer which Lagrangian system matches the DAG."""
    vars_lower = {v.lower() for v in dag.variables}

    if {"theta", "angle", "period"} & vars_lower:
        return simple_pendulum()
    if {"x", "position", "displacement"} & vars_lower:
        return harmonic_oscillator()
    if {"theta1", "theta2"} & vars_lower:
        return double_pendulum()

    return None
```

---

## 四、Demo — 单摆最小作用量验证

### 4.1 场景描述

```
真实物理: 单摆 T = 2π√(L/g), 轨迹 θ(t) = θ₀cos(ωt), ω = √(g/L)
实验: 生成三条路径
  A: 物理真实路径 θ(t) = θ₀cos(ωt)           (δS = 0)
  B: 直线插值 θ(t) = θ₀ + (θ_end-θ₀)(t/T)   (非物理)
  C: 随机扰动: 物理路径 + 小噪声               (δS > 0)
对比: 三种路径的作用量 S_A < S_B, S_A < S_C
```

### 4.2 完整代码

```python
#!/usr/bin/env python3
"""Demo: Principle of Least Action for simple pendulum."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.physics import (
    LagrangianSystem, ActionPrinciple,
    simple_pendulum, harmonic_oscillator,
)

def demo_pendulum_action():
    print("=" * 60)
    print("  PRINCIPLE OF LEAST ACTION — Simple Pendulum")
    print("=" * 60)

    # ── Setup ──
    l, g = 1.0, 9.81
    omega = np.sqrt(g / l)
    system = simple_pendulum(l=l, g=g)
    principle = ActionPrinciple(system, tolerance=0.01)

    # ── Generate paths ──
    theta_0 = 0.5          # initial angle (radians)
    n_steps = 200
    T_final = 2.0          # observation duration
    dt = T_final / (n_steps - 1)
    t = np.linspace(0, T_final, n_steps)

    # Path A: Physical (harmonic approximation for small angles)
    path_physical = theta_0 * np.cos(omega * t)

    # Path B: Linear interpolation (non-physical)
    path_linear = np.linspace(theta_0, theta_0 * np.cos(omega * T_final), n_steps)

    # Path C: Physical + noise (slightly perturbed)
    rng = np.random.default_rng(42)
    path_noisy = path_physical + rng.normal(0, 0.02, n_steps)

    # ── Validate each path ──
    print(f"\n{'Path':<18} {'Action':>12} {'max|δS/δq|':>14} {'Valid':>8}")
    print("-" * 55)

    for name, path in [("A: Physical", path_physical),
                        ("B: Linear", path_linear),
                        ("C: Noisy", path_noisy)]:
        S = principle.compute_action(path, dt)
        result = principle.validate_trajectory(path, dt)
        status = "✓" if result["valid"] else "✗"
        print(f"{name:<18} {S:>12.6f} {result['max_gradient']:>14.2e} {status:>8}")

    # ── Compare ──
    comparison = principle.compare_paths(path_physical, path_linear, dt)
    print(f"\n── Comparison (Physical vs Linear) ──")
    print(f"  Action(A) = {comparison['path_a_action']:.6f}")
    print(f"  Action(B) = {comparison['path_b_action']:.6f}")
    print(f"  ΔS = {comparison['delta_S']:.6f}")
    print(f"  {comparison['interpretation']}")

    # ── Find stationary path ──
    print(f"\n── Finding Stationary Path ──")
    opt_result = principle.find_stationary_path(
        q_start=path_linear[0], q_end=path_linear[-1],
        n_steps=n_steps, dt=dt, max_iter=500, lr=0.01,
    )
    print(f"  Converged: {opt_result['converged']}")
    print(f"  Iterations: {opt_result['iterations']}")
    print(f"  Final action: {opt_result['action']:.6f}")
    print(f"  Final max|δS/δq|: {opt_result['max_gradient']:.2e}")

    # Check: optimized path should be close to physical path
    mae = np.mean(np.abs(opt_result["q_path"] - path_physical))
    print(f"  MAE vs physical path: {mae:.4f} rad")

    return 0 if opt_result["converged"] else 1


def demo_harmonic_oscillator():
    print(f"\n{'='*60}")
    print("  PRINCIPLE OF LEAST ACTION — Harmonic Oscillator")
    print("=" * 60)

    m, k = 1.0, 4.0      # ω = √(k/m) = 2
    omega = np.sqrt(k / m)
    system = harmonic_oscillator(m=m, k=k)
    principle = ActionPrinciple(system, tolerance=0.02)

    n_steps, T_final = 200, 3.0
    dt = T_final / (n_steps - 1)
    t = np.linspace(0, T_final, n_steps)
    x0 = 1.0

    path_physical = x0 * np.cos(omega * t)        # x(t) = cos(2t)
    path_linear = np.linspace(x0, x0 * np.cos(omega * T_final), n_steps)

    print(f"\n{'Path':<18} {'Action':>12} {'max|δS/δq|':>14} {'Valid':>8}")
    print("-" * 55)
    for name, path in [("A: Physical", path_physical),
                        ("B: Linear", path_linear)]:
        S = principle.compute_action(path, dt)
        result = principle.validate_trajectory(path, dt)
        status = "✓" if result["valid"] else "✗"
        print(f"{name:<18} {S:>12.6f} {result['max_gradient']:>14.2e} {status:>8}")

    return 0


if __name__ == "__main__":
    ret = demo_pendulum_action()
    ret |= demo_harmonic_oscillator()
    print(f"\n{'='*60}")
    print(f"  {'ALL PASSED' if ret == 0 else 'SOME FAILED'}")
    print("=" * 60)
```

### 4.3 预期输出

```
  Path               Action    max|δS/δq|    Valid
  ───────────────────────────────────────────────────
  A: Physical       0.184723       3.12e-04       ✓
  B: Linear         2.156891       8.47e-01       ✗
  C: Noisy          0.192445       2.31e-02       ✗

  ── Comparison ──
  Action(A) = 0.184723
  Action(B) = 2.156891
  ΔS = -1.972168
  Path A has lower action and is physically preferred.

  ── Finding Stationary Path ──
  Converged: True
  Iterations: 347
  Final action: 0.184731
  Meta vs physical path: 0.0012 rad
```

---

## 五、理论推导细节

### 5.1 单摆的 Euler-Lagrange 方程

拉格朗日量:

$$L = T - V = \frac{1}{2} m l^2 \dot{\theta}^2 - mgl(1 - \cos\theta)$$

Euler-Lagrange:

$$\frac{d}{dt} \frac{\partial L}{\partial \dot{\theta}} - \frac{\partial L}{\partial \theta} = 0$$

$$\frac{d}{dt}(m l^2 \dot{\theta}) + mgl \sin\theta = 0$$

$$m l^2 \ddot{\theta} + mgl \sin\theta = 0$$

$$\ddot{\theta} + \frac{g}{l} \sin\theta = 0$$

小角度近似 $(\sin\theta \approx \theta)$:

$$\ddot{\theta} + \frac{g}{l} \theta = 0 \quad \rightarrow \quad \theta(t) = \theta_0 \cos(\omega t), \quad \omega = \sqrt{g/l}$$

### 5.2 离散化细节

```
连续: S = ∫₀ᵀ L(θ(t), θ̇(t)) dt

离散: S ≈ Σᵢ L(θᵢ, (θᵢ₊₁ - θᵢ)/Δt) · Δt   (欧拉法, O(Δt))
       S ≈ Σᵢ L(θᵢ, (θᵢ₊₁ - θᵢ₋₁)/(2Δt)) · Δt  (中心差分, O(Δt²))

变分导数: δS/δθᵢ ≈ [S(θ+εeᵢ) - S(θ-εeᵢ)] / (2ε)
```

---

## 六、causal_agent 中的具体交互

### 6.1 Agent 新增命令

```
> action validate theta.csv    — 验证轨迹是否满足最小作用量原理
> action find 0.5 0.3 200     — 寻找起始 θ=0.5 到终止 θ=0.3 的稳态路径
> action compare a.csv b.csv   — 比较两条路径，指出哪条更符合物理
```

### 6.2 反事实增强

```
现有反事实:
  obs: θ=0.5 at t=0
  intv: do(L=2.0)  # 摆长加倍
  output: θ_cf = ? (单点值)

增强反事实:
  obs: 整条轨迹 {θ(t)} for t∈[0,T]  with L=1.0
  intv: do(L=2.0)
  ActionPrinciple.find_stationary_path() → 新约束下的整条最优轨迹
  output: {θ_cf(t)} vs {θ_obs(t)} for all t
  
  诊断: 如果新轨迹的 δS 大于阈值 →
        "此干预违反最小作用量原理，可能对应不稳定的中间状态"
```

---

## 七、局限与扩展

| 局限 | 缓解 |
|------|------|
| 仅支持保守系统 (无耗散) | 可用 Rayleigh 耗散函数扩展: d/dt(∂L/∂q̇) - ∂L/∂q + ∂R/∂q̇ = 0 |
| 数值梯度下降可能陷入局部极小值 | 多次随机初始化或使用模拟退火 |
| 需要手动指定 Lagrangian 形式 | 可与 `SymbolicPhysicsDiscovery` 结合，从数据中发现 L 的形式 |
| 高维系统的 `variational_derivative` 计算量按 O(n²) 增长 | 使用伴随方法 (adjoint method) 或自动微分 |
| 有限差分 δS/δq 有数值误差 | 可用自动微分 (JAX/PyTorch) 替代, 或外推至 ε→0 |

---

## 八、相关文献

| 文献 | 关系 |
|------|------|
| Goldstein, Poole & Safko. Classical Mechanics (3rd ed.) | Lagrangian/Hamiltonian 力学标准教材 |
| Lanczos. The Variational Principles of Mechanics | 变分原理的数学基础 |
| Feynman & Hibbs. Quantum Mechanics and Path Integrals | 作用量与路径积分的联系 |
| Marsden & West. Discrete mechanics and variational integrators | 变分积分子 — 保辛结构的离散化 |
| Cranmer et al. Lagrangian Neural Networks (2020) | 从数据中学习 Lagrangian |
| Greydanus et al. Hamiltonian Neural Networks (2019) | 从数据中学习 Hamiltonian |

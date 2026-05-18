# 物理规律扩展指南

> 适用版本: v0.9.7 | 核心模块: `core/physics.py`

---

## 一、设计理念

智能体通过 `PhysicsLibrary` 管理物理定律。每条定律不仅包含数学公式，还编码了**因果方向的强制约束**——物理定律是因果推断的"第一性原理"。

```
因果推断:  谁影响谁？    →  CausalDAG 提供骨架
物理定律:  影响必须是怎样的？ → PhysicsLaw 提供血肉
```

---

## 二、PhysicsLaw 结构

```python
PhysicsLaw(
    name="定律名称",                  # 唯一标识符
    domain="领域",                    # mechanics / thermo / em / fluids / ...
    latex=r"$LaTeX公式$",            # 显示用
    
    # 因果图约束
    inputs=["原因1", "原因2"],       # 输入变量（原因）
    outputs=["结果1"],                # 输出变量（结果）
    parameters=["参数"],              # 常数参数（可选）
    
    # 约束类型
    constraint_type=ConstraintType.SCM_EQUATION,  # 见下文
    
    # 公式（Python可执行）
    formula="lambda x,y: x / y",     # 结构方程
    
    # 因果方向控制
    causal_direction=[("原因","结果")],     # 强制因果方向
    forbidden_directions=[("结果","原因")], # 禁止反向因果
    required_parents={"变量":["必须的父节点"]}, # 守恒律要求
    
    tolerance=0.05,                   # 验证容差
)
```

### 五种约束类型

| 类型 | 作用 | 示例 |
|------|------|------|
| `SCM_EQUATION` | 替换结构方程 | $a=F/m$ |
| `CONSERVATION` | 验证守恒律 | $\sum p_{before}=\sum p_{after}$ |
| `DAG_EDGE` | 强制/禁止因果边 | $F\to a$ (禁止 $a\to F$) |
| `BOUNDARY` | 取值范围约束 | $0\leq\eta\leq1$ |
| `SYMMETRY` | 对称性约束 | 平移不变性, 旋转不变性 |

---

## 三、三步扩展流程

### 步骤 1: 定义定律

```python
from core.physics import PhysicsLibrary, PhysicsLaw, ConstraintType

physics = PhysicsLibrary()
```

**例 1: SCM 方程类型（最常见）**

```python
# 万有引力: F = Gm₁m₂/r²
physics.register(PhysicsLaw(
    name="newton_gravity",
    domain="mechanics",
    latex=r"$F = G\frac{m_1 m_2}{r^2}$",
    inputs=["Mass1", "Mass2", "Distance"],
    outputs=["GravitationalForce"],
    parameters=["G"],
    constraint_type=ConstraintType.SCM_EQUATION,
    formula="lambda m1,m2,r, G=6.674e-11: G * m1 * m2 / max(r**2, 1e-20)",
    causal_direction=[
        ("Mass1", "GravitationalForce"),
        ("Mass2", "GravitationalForce"),
    ],
    forbidden_directions=[
        ("GravitationalForce", "Mass1"),  # 力不能改变质量
    ],
))
```

**例 2: 守恒律类型**

```python
# 质量守恒: m_in = m_out
physics.register(PhysicsLaw(
    name="mass_conservation",
    domain="mechanics",
    latex=r"$\sum m_{in} = \sum m_{out}$",
    inputs=["Mass_in", "Mass_out"],
    outputs=[],
    constraint_type=ConstraintType.CONSERVATION,
    formula="np.mean(np.abs(data['Mass_in'] - data['Mass_out']))",
    tolerance=0.01,
))
```

**例 3: DAG 边约束类型**

```python
# 时间之箭: 未来事件不能导致过去事件
physics.register(PhysicsLaw(
    name="time_arrow",
    domain="universal",
    latex=r"$t_{cause} < t_{effect}$",
    inputs=[],
    outputs=[],
    constraint_type=ConstraintType.DAG_EDGE,
    causal_direction=[],  # 由下方 forbidden 实现
    forbidden_directions=[
        ("FutureVar", "PastVar"),  # 具体变量名
    ],
))
```

### 步骤 2: 接入流水线

```python
from core.physics import physics_causal_pipeline

result = physics_causal_pipeline(
    data, var_names,
    treatment="Treatment", outcome="Outcome",
    domain="mechanics",   # 可选: 仅加载特定领域
)

# 查看结果
print(result["physics_equations"])   # 被物理定律替换的方程
print(result["constraints"])         # 因果图约束
print(result["summary"])             # 完整分析报告
```

### 步骤 3: 验证

```python
# 生成测试数据
import numpy as np
rng = np.random.default_rng(42)
n = 1000
Mass1 = rng.uniform(1, 100, n)
Mass2 = rng.uniform(1, 100, n)
Distance = rng.uniform(0.1, 10, n)
G = 6.674e-11
Force = G * Mass1 * Mass2 / Distance**2 + rng.normal(0, 1e-12, n)

data = np.column_stack([Mass1, Mass2, Distance, Force])
var_names = ["Mass1", "Mass2", "Distance", "GravitationalForce"]

result = physics_causal_pipeline(data, var_names,
                                  "Mass1", "GravitationalForce")

# 验证定律被识别
assert "GravitationalForce" in result["physics_equations"]
print("✓ 万有引力定律已被自动匹配")
```

---

## 四、formula 参数编写规则

`formula` 是一个 Python lambda 字符串，参数顺序对应 `inputs` 和 `parameters` 的顺序。

```python
# 规则 1: 先 inputs, 后 parameters
PhysicsLaw(
    inputs=["X", "Y"],
    parameters=["a", "b"],
    formula="lambda x,y, a=1.0, b=2.0: a * x + b * y"
    #         ^ inputs顺序  ^ params with defaults
)

# 规则 2: 用 max(x, epsilon) 防止除零
formula="lambda a,b: a / max(b, 0.001)"

# 规则 3: 可以用 np 函数
formula="lambda x: np.sin(x) + np.exp(-x**2)"

# 规则 4: 常量放 lambda 体内
formula="lambda t: 2 * 3.14159 * (t / 9.81)**0.5"  # 2π√(t/g)
```

---

## 五、领域组织

建议按以下领域组织定律：

```
physics.register(...)  # domain="mechanics"
  - 牛顿定律 (F=ma)
  - 动能定理 (E_k=½mv²)
  - 动量守恒
  - 能量守恒
  - 万有引力
  - 胡克定律 (F=-kx)
  - 单摆周期

physics.register(...)  # domain="electromagnetism"
  - 欧姆定律 (V=IR)
  - 焦耳定律 (P=I²R)
  - 麦克斯韦方程组（分量形式）

physics.register(...)  # domain="thermodynamics"
  - 理想气体定律 (PV=nRT)
  - 第二热力学定律 (ΔS≥0)
  - 热传导方程

physics.register(...)  # domain="fluids"
  - 伯努利方程
  - 连续性方程 (A₁v₁=A₂v₂)

physics.register(...)  # domain="quantum"
  - 不确定性原理 (ΔxΔp≥ℏ/2)
  - 薛定谔方程（定态）
  - 光电效应 (E=hν-W)
```

---

## 六、完整代码模板

```python
"""my_physics_laws.py — 自定义物理定律库"""

from core.physics import PhysicsLibrary, PhysicsLaw, ConstraintType

def build_my_library() -> PhysicsLibrary:
    lib = PhysicsLibrary()

    # ===== 你的定律在这里 =====

    lib.register(PhysicsLaw(
        name="my_law",
        domain="my_domain",
        latex=r"$Y = \alpha X + \beta$",
        inputs=["X"],
        outputs=["Y"],
        parameters=["alpha", "beta"],
        constraint_type=ConstraintType.SCM_EQUATION,
        formula="lambda x, alpha=1.0, beta=0.0: alpha * x + beta",
        causal_direction=[("X", "Y")],
        forbidden_directions=[("Y", "X")],
    ))

    # ==========================

    return lib

# 使用
if __name__ == "__main__":
    lib = build_my_library()
    print(lib.summary())

    # 接入物理因果流水线
    from core.physics import physics_causal_pipeline
    import numpy as np

    rng = np.random.default_rng(42)
    X = rng.uniform(0, 10, 500)
    Y = 2.0 * X + 1.0 + rng.normal(0, 0.1, 500)
    data = np.column_stack([X, Y])

    result = physics_causal_pipeline(data, ["X", "Y"], "X", "Y")
    print(result["summary"])
```

---

## 七、常见问题

**Q: 定律的名称有什么要求？**
A: 必须唯一。建议用 `domain_shortname` 格式，如 `mechanics_newton2`。

**Q: formula 中能使用 numpy 吗？**
A: 可以。`np.sin`, `np.exp`, `np.sqrt` 等都可用。但需要 `import numpy as np`。

**Q: 如何表示"没有公式只有方向约束"？**
A: 设置 `constraint_type=ConstraintType.DAG_EDGE`，`formula=None`。

**Q: 如何调试 formula？**
A: 手动测试：
```python
fn = eval("lambda x,y: x / max(y, 0.001)")
print(fn(10, 2))  # 应该输出 5.0
```

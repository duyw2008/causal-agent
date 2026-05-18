# Causal Agent — 学习文档

> 版本: v0.9.10 | 日期: 2026-05-18
> 定位: 从零理解因果推断智能体的架构、算法与数据生成

---

## 目录

1. [核心理念](#一核心理念)
2. [系统架构](#二系统架构)
3. [因果图模型](#三因果图模型)
4. [因果发现算法](#四因果发现算法)
5. [因果效应识别](#五因果效应识别)
6. [效应估计器](#六效应估计器)
7. [反事实推理](#七反事实推理)
8. [因果中介分析](#八因果中介分析)
9. [敏感性分析](#九敏感性分析)
10. [物理规律约束](#十物理规律约束)
11. [LLM 自然语言接口](#十一llm-自然语言接口)
12. [数据生成方法](#十二数据生成方法)
13. [运行流程](#十三运行流程)

---

## 一、核心理念

### 1.1 Agent 解决什么问题

```
传统 ML:  看到 "冰淇淋销量" 和 "溺水死亡" 都高 → 预测它们相关
Causal Agent: 知道它们都因为 "天气热" → 降低冰淇淋销量不会减少溺水

本质区别:  观测关联 ≠ 因果效应
          P(Y|X)   ≠  P(Y|do(X))
```

### 1.2 设计哲学：结构与参数分离

```
Step 1: 先确定 "谁影响谁" (因果图 / CausalDAG)
          ↓
Step 2: 再量化 "影响多大" (因果效应 / ATE)

原因: 因果结构是定性的、稳定的；
      效应大小是定量的、随数据变化的。
      把两者分开可以避免 "垃圾进垃圾出"。
```

### 1.3 三层因果推理（Pearl 因果阶梯）

```
Layer 3 — 反事实 (Counterfactual)
  "如果这个人当年没上大学，收入会是多少？"
  需要: SCM + 外生噪声回溯
  
Layer 2 — 干预 (Intervention)
  "如果所有人上大学，平均收入会增加多少？"
  需要: do-calculus + 效应估计
  
Layer 1 — 关联 (Association)
  "教育水平和收入相关吗？"
  需要: 纯统计 (相关性)
  
Agent 同时覆盖三层。
```

---

## 二、系统架构

### 2.1 模块全景

```
用户输入
    │
    ├── 自然语言 (LLM 模式) ──→ ask "吸烟会导致肺癌吗"
    │                                │
    │      ┌─────────────────────────┘
    │      ▼
    ├── llm_client.py ──→ 提取因果图 JSON
    │      │
    │      ▼
    ├── CausalDAG (graph.py)
    │      │  d-separation, 拓扑排序, 祖先/后代
    │      ▼
    ├── identification.py
    │      │  back-door / front-door / IV / do-calculus
    │      ▼
    ├── estimation.py + modern.py
    │      │  11 种估计器, 自动诊断, 自动方法选择
    │      ▼
    ├── sensitivity.py ──→ Rosenbaum bounds + E-value
    │      │
    │      ▼
    └── llm_client.py ──→ 中文自然语言解读
```

### 2.2 13 个核心模块

| 模块 | 文件 | 职责 |
|------|------|------|
| 因果图 | `core/graph.py` | DAG 数据结构、d-separation、拓扑排序 |
| 效应识别 | `core/identification.py` | back-door、front-door、IV、do-calculus |
| 结构方程 | `core/scm.py` | 线性/非线性 SCM、do() 干预、反事实 |
| 因果发现 | `core/discovery.py` | PC、FCI、GES、自举置信度 |
| 效应估计 | `core/estimation.py` | Linear、PSM、IPW、DR、Stratified |
| 现代方法 | `core/modern.py` | DML、S/T/X-Learner、CausalForest、do-why |
| 敏感性 | `core/sensitivity.py` | Rosenbaum 界限、E-value |
| 中介分析 | `core/mediation.py` | NDE、NIE、CDE、Baron-Kenny、Pearl 反事实 |
| 物理引擎 | `core/physics.py` | 15 条定律、变分原理、Lagrangian 力学 |
| 时间序列 | `core/ts_discovery.py` | Granger、TS-PC、PCMCI-lite |
| LLM 客户端 | `core/llm_client.py` | DeepSeek API、因果图提取、结果解读 |
| 自然语言 | `nlp/parser.py` | 规则+模板解析器 |
| 可视化 | `core/visualization.py` | ASCII、DOT、Mermaid、PNG |

---

## 三、因果图模型

### 3.1 CausalDAG — 数据结构

```python
from core.graph import CausalDAG

# 创建因果图: X 影响 Y, Z 同时影响 X 和 Y (混淆)
dag = CausalDAG(
    variables=["X", "Y", "Z"],
    edges=[("Z", "X"), ("Z", "Y"), ("X", "Y")]
)
```

```
    Z
   ↙ ↘
  X → Y

语义: Z 是混淆因子 (confounder)
      X→Y 是我们关心的因果路径
```

### 3.2 核心操作

| 操作 | 含义 | 示例 |
|------|------|------|
| `dag.parents("Y")` | Y 的直接原因 | `{"X", "Z"}` |
| `dag.children("Z")` | Z 的直接影响 | `{"X", "Y"}` |
| `dag.ancestors("Y")` | Y 的所有祖先 | `{"X", "Z"}` |
| `dag.is_d_separated(X,Y,Z)` | X 和 Y 在给定 Z 下是否独立？ | 混淆图中 True |
| `dag.topological_order()` | 因果序 | `["Z", "X", "Y"]` |

### 3.3 d-separation — 条件独立性的图准则

```
三种连接模式:

1. 链 (Chain):     X → M → Y
   X 和 Y 相关。给定 M 后, X ⊥ Y | M  ✓ (阻塞)

2. 叉 (Fork):      X ← Z → Y
   X 和 Y 相关。给定 Z 后, X ⊥ Y | Z  ✓ (阻塞)

3. 对撞 (Collider): X → Z ← Y
   X 和 Y 独立。给定 Z 后, X ⊥̸ Y | Z  ✗ (打开!)

关键直觉:
  - 非对撞节点在条件集中 → 阻塞路径
  - 对撞节点在条件集中 → 打开路径
  - 对撞节点的后代在条件集中 → 也打开路径
```

---

## 四、因果发现算法

### 4.1 方法论：从数据到 SCM 的两步走

```
原始数据 (CSV)
      │
      ▼
┌─────────────────┐
│ Step 1: 结构发现  │  ← 因果发现算法 (PC / FCI / GES)
│ "谁影响谁？"     │     输出: DAG 或 PAG
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Step 2: 参数估计  │  ← 回归 / 最大似然
│ "影响多大？"     │     输出: 结构方程 f_i
└────────┬────────┘
         │
         ▼
    完整 SCM
```

**为什么数据不能直接说出因果关系，但可以泄露因果方向？**

```
原始数据只告诉我们:
  所有变量的联合分布 P(X, Y, Z, ...)

但三种因果结构在数据中有不同的「条件独立性指纹」:

  Fork (混淆):    X ← Z → Y
    指纹: X ⊥ Y | Z  (给定 Z 后 X 和 Y 独立)
    → 如果控制了 Z，X 和 Y 不相关

  Chain (中介):   X → M → Y
    指纹: X ⊥ Y | M  (给定 M 后 X 和 Y 独立)
    → 如果控制了 M，X 对 Y 没有额外的预测能力

  Collider (对撞): X → Z ← Y
    指纹: X ⊥ Y      (无条件时 X 和 Y 独立)
          X ⊥̸ Y | Z  (给定 Z 后反而不独立!)
    → 这是因果方向的关键信号——只有对撞结构会产生这种模式

PC 算法的核心思想:
  穷举所有变量对的所有可能的条件集，
  用条件独立性检验来区分这三种结构。
```

**PC 算法从条件独立性到因果图的过程：**

```
输入: 500 行 × 3 列的 CSV
  X     Y     Z
 1.2   3.1   0.5
 0.8   2.7   1.2
  ...

Step 1 — 骨架学习:
  从完全图开始 (所有变量两两相连)。
  对每对变量 (A, B)，测试:
    是否存在某个变量集 S，使得 A ⊥ B | S？

  例: X—Y: 测试 X ⊥ Y | ∅  → 相关 → 保留边
           测试 X ⊥ Y | Z  → 独立 → 删除边 X—Y
     → X 和 Y 原本的相关是由 Z 驱动的 (Fork)

Step 2 — v-structure 定向:
  找到 X—Z—Y 模式，其中 X 和 Y 不相邻。
  检查: Z 是否在 X 和 Y 的分离集中？
    不在 → Z 是对撞节点 → X → Z ← Y
    在   → Z 不是对撞节点，方向暂不确定

Step 3 — Meek 规则传播:
  利用已定向的边按逻辑规则推断其余边的方向。
  例: 已知 X→Z 且 Z—Y 且 X 和 Y 不相邻
      → 则 Z—Y 不能是 Z←Y (会产生新对撞与 Step 2 矛盾)
      → 因此 Z→Y
```

**从 DAG 骨架到完整 SCM：**

```
有了 DAG 骨架后，按拓扑序为每个节点拟合方程:

  已知 DAG:
    Ability ──→ Education ──→ Income
      │                        ↗
      └────────────────────────┘

  按拓扑序:
    Education = γ · Ability + U_Education
    Income    = β₁·Education + β₂·Ability + U_Income

  方法: 以每个变量的父节点为自变量做普通回归。
  SCM 本质上是「DAG 结构 + 每个节点一个回归方程」。
```

**三种因果发现算法对比：**

```
输入:  CSV 数据 (n_samples × n_variables)
输出:  CausalDAG (谁影响谁？)

三种算法，按假设从强到弱:
```

### 4.2 PC 算法 (约束方法)

```
假设: 无隐变量 (causal sufficiency)
原理: 条件独立性检验 → 骨架学习 → v-structure 定向 → Meek 规则
复杂度: O(n²) → O(nᵏ) per CI test

步骤:
  1. 从完全图开始
  2. 对每条边 X—Y, 测试 X ⊥ Y | S (S 大小从 0 到 k)
     如果独立 → 删除边
  3. v-structure 定向: X—Z—Y, X 和 Y 不相邻,
     如果 Z 不在 X 和 Y 的分离集中 → X→Z←Y
  4. Meek 规则 R1-R3 传播方向

使用:
  > discover data.csv pc
  > discover data.csv pc --alpha=0.01  (更保守)
```

### 4.3 FCI 算法 (允许隐变量)

```
假设: 允许未测量的混淆变量
原理: PC 骨架 + 额外 CI 检验 → R1-R7 定向规则 (Zhang 2008)
输出: PAG (Partial Ancestral Graph, 7 种边类型)

边类型:
  →   (directed — 确定因果方向)
  ◦→  (ancestral — A 是 B 的祖先, 但可能间接)
  ◦—◦ (undetermined — 关系不确定)
  ↔   (confounded — 存在未测量混淆!)

使用:
  > discover data.csv fci
```

### 4.4 GES 算法 (评分方法, 三阶段)

```
假设: 无隐变量, 但用 BIC 评分而非 CI 检验
原理: 贪心搜索使 BIC 最优的图结构
复杂度: O(|V|³) per iteration

三阶段:
  Phase 1 (Forward):  空图 → 贪心加边 (选 BIC 提升最大的)
  Phase 2 (Backward): 贪心删边 (选 BIC 提升最大的)
  Phase 3 (CI Pruning): 条件独立性剪枝
    — 消除 collider X→Z←Y 中 BIC 误导的 X→Y 多余边

BIC 评分:
  BIC(G) = −(n/2)Σ log(2π·σ̂² + 1) − (|PA|+2)/2 · log n
  
  第一项: 拟合优度 (残差越小越好)
  第二项: 复杂度惩罚 (边越多惩罚越重)

使用:
  > discover data.csv ges
```

### 4.5 自举置信度

```
问题: 单次因果发现的边可能不稳定
解决: Bootstrap — 重采样 B 次, 统计每条边出现的频率

confidence(X→Y) = count(X→Y in graph_b) / B

使用:
  > discover data.csv pc --bootstrap=100
  输出: X→Y: 0.87 ████████████████░░░  (高置信)
        X→Z: 0.42 ████████░░░░░░░░░░░░  (低置信, 可能假阳性)
```

### 4.6 现实中的三大难点

**难点 1 — Markov 等价类：结构不唯一**

```
         X                   X
        ↗ ↘                ↗
       Z   Y     ≡     Z → Y

这两张图产生完全相同的数据分布和条件独立性模式。
纯观测数据无法区分它们。

PC 输出 CPDAG (部分有向图) 而非完整 DAG:
  等价类中的无向边需要「干预实验」或「领域知识」来定向。

FCI 更诚实: 输出 PAG，明确标注
  ◦→  (可能是祖先，但不确定是否直接)
  ◦—◦ (完全不确定)
  ↔   (存在隐变量混淆)
```

**难点 2 — 因果充分性假设**

```
PC 和 GES 假设「没有未观测的混淆变量」。

如果假设被违反:
  
  真实图:     [U]   (U 未被测量)
             ↙ ↘
            X   Y
            ↘ ↙
              Z
              
  PC 发现:   X ↔ Y  (FCI 会输出 bidirected edge)
  
  FCI 通过 Possible-D-SEP 额外检验来检测隐变量模式，
  但也不是万能的。某些隐变量结构是无法从观测数据中识别的。
```

**难点 3 — 样本量限制**

```
条件独立性检验需要足够的样本:
  - 变量越多 → 条件集越大 → 需要的样本指数增长
  - 对小样本: 降低 α (如 0.01) 减少假阳性
  - 对大样本: 提高 α (如 0.1) 增加统计效力

Bootstrap 量化不确定性:
  > discover data.csv pc --bootstrap=100

  边 X→Y: 置信度 0.87  ████████████████░░░  (稳定)
  边 X→Z: 置信度 0.42  ████████░░░░░░░░░░░░  (可能是假阳性)
```

### 4.7 Agent 中从数据到因果结论的完整流水线

```
> discover data.csv pc --bootstrap=50
  → 自动发现因果结构 + 每条边的置信度评估

> effect X Y
  → 识别因果效应 (back-door / do-calculus)
  → 自动诊断数据质量 (正态性 + 重叠性)
  → 自动选择最佳估计器
  → ATE 输出 + 95% 置信区间

> whatif X=0 Y
  → 基于已发现的 SCM 做干预预测
  → do(X=0) 后的 Y 期望值

> sensitivity X Y
  → Rosenbaum bounds + E-value
  → 「需要多强的未测量混淆才能推翻这个结论？」

这一条流水线实现了:
  原始 CSV → 因果结构 → 效应识别 → 效应估计 → 反事实 → 敏感性检验
```

---

## 五、因果效应识别

### 5.1 四种识别策略

```
问题: P(Y|do(X)) 能从观测数据中估计吗？

策略 1 — Back-door (后门调整): 最常用
  条件: 存在调整集 Z 阻断 X 和 Y 之间的所有后门路径
  公式: P(y|do(x)) = Σ_z P(y|x,z) P(z)
  示例: Z→X, Z→Y, X→Y → 调整 Z

策略 2 — Front-door (前门调整): 有中介时
  条件: 存在 M 截获所有因果路径, 且 X→M 无后门, M→Y 的后门被 X 阻塞
  公式: P(y|do(x)) = Σ_m P(m|x) Σ_x' P(y|x',m) P(x')
  示例: X→M→Y (有未测量混淆时)

策略 3 — Instrumental Variable (工具变量): 有工具时
  条件: Z 影响 X, Z 只通过 X 影响 Y, Z 无混淆
  公式: ATE = Cov(Y,Z) / Cov(X,Z)

策略 4 — do-calculus (v0.9.4): 穷举搜索
  方法: 穷举所有可能的调整集 W, 在 G_{X̄} 中验证 Y ⊥ Z | X,W
  使用: 前三种策略失败时自动 fallback
```

### 5.2 do-calculus 三条规则 (Pearl)

```
Rule 1: P(y|do(x),z,w) = P(y|do(x),w)
         if Y ⊥ Z | X,W in G_{X̄}
         含义: 可以删除无关的观测条件

Rule 2: P(y|do(x),do(z),w) = P(y|do(x),z,w)
         if Y ⊥ Z | X,W in G_{X̄,Z̲}
         含义: 可以用观测替代干预

Rule 3: P(y|do(x),do(z),w) = P(y|do(x),w)
         if Y ⊥ Z | X,W in G_{X̄,Z(W)̅}
         含义: 可以删除无关的干预
```

---

## 六、效应估计器

### 6.1 问题形式化

```
给定: 数据 D = {(Y_i, T_i, Z_i)}_{i=1}^n
      T = treatment (处理变量)
      Y = outcome  (结果变量)
      Z = 调整集   (来自 identification)
目标: ATE = E[Y|do(T=1)] - E[Y|do(T=0)]
```

### 6.2 经典估计器 (5 种)

| 估计器 | 原理 | 适用场景 |
|--------|------|---------|
| **Linear** | OLS: Y ~ T + Z | 线性假设成立, 重叠好 |
| **PSM** | 倾向得分逻辑回归 + 最近邻匹配 | 二值处理, 重叠好 |
| **IPW** | 1/PropensityScore 加权 | 连续或二值处理 |
| **DR** | PSM + OLS 结合, 双保险 | 模型误设时仍一致 |
| **Stratified** | 按 Z 分层估计, Mantel-Haenszel | 离散混杂因子 |

### 6.3 现代估计器 (6 种)

| 估计器 | 原理 | 输出 |
|--------|------|------|
| **Double ML** | 交叉拟合 + Neyman 正交 | ATE ± CI |
| **S-learner** | 单一模型, T 作为特征 | CATE (个体效应) |
| **T-learner** | 处理组/对照组分别建模 | CATE |
| **X-learner** | 处理组建模 + 虚拟对照组 | CATE |
| **CausalForest** | 树模型 + 诚实估计 | CATE |
| **do-why** | Microsoft do-why 库 | ATE |

### 6.4 自动诊断与自动方法选择 (v0.9.6)

```
Agent 自动执行:
  1. 残差正态性检验 (skewness + kurtosis)
  2. 协变量重叠性检验 (SMD — 标准化均值差)
  3. 根据诊断结果自动选择:
  
     linearity ✓ + overlap ✓  → linear (最高效)
     overlap ~                → PSM (匹配解决重叠问题)
     linearity ~              → IPW (不需要线性假设)
     linearity ✗ + overlap ✗  → DR (双重鲁棒, 最安全)
```

---

## 七、反事实推理

### 7.1 三步反事实 (Pearl)

```
问题: "如果这个人当年没上大学, 收入会是多少？"

已知: 他上了大学, 收入 8 万/年
反事实: 他没上大学会怎样？

Step 1 — Abduction (溯因):
  从观测反推外生噪声:
  u = Y_obs - f(parents_obs)
  
Step 2 — Action (行动):
  施加干预: do(Education=0)
  修改结构方程: Income = f(Education=0, ...)
  
Step 3 — Prediction (预测):
  用修改后的方程 + 同样的外生噪声 u 计算结果:
  Y_cf = f(Education=0, ...) + u
```

### 7.2 线性和非线性 SCM

```python
# 线性 SCM
scm = linear_scm(dag, {"Y": {"X": 2.0}}, noise_std=0.1)
cf = scm.counterfactual({"X": 1.0, "Y": 2.5}, {"X": 0.0}, "Y")
# → Y_cf = 0.5 (噪声被保持了)

# 非线性 SCM
scm = nonlinear_scm(dag, {
    "Y": nonlinear_eq(lambda x: x**2 + 3*x, ["X"])
})
cf = scm.counterfactual({"X": 2.0, "Y": 10.5}, {"X": 0.0}, "Y")
# → Y_cf = 0.5
```

### 7.3 线性 SCM 的数据生成原理

```python
# SCM 如何生成数据:

# 定义结构方程:
#   X = noise_X
#   Y = 2.0 * X + noise_Y

# 按拓扑序计算:
scm.sample(1000)

# 等价于:
for i in range(1000):
    x = np.random.normal(0, 1)       # 外生变量先采样
    y = 2.0 * x + np.random.normal(0, 0.1)  # 内生变量后计算

# 关键: 顺序必须按因果方向
# 原因在前, 结果在后
```

### 7.4 核心洞察：为什么 SCM 只需要一套方程？

```
████████████████████████████████████████████████████████████
  结构方程描述的是「变量如何产生」，而非「特定取值下的分布」。
  因此干预只改变输入值，方程本身不需要重新拟合。
████████████████████████████████████████████████████████████
```

**直觉例子 — 张三的教育-收入反事实：**

```
已知:
  SCM 中:  Income = β · Education + γ · Ability + U_张三
  
  实际观测:  Education = 16  →  Income = 3万/年
  反事实问:  如果 Education = 12, Income 会变吗？

SCM 的反事实推理:
  
  Step 1 — Abduction (溯因):
    从观测事实反推张三的「不可观测因子」U_张三:
    U_张三 = 3万 − β·16 − γ·Ability_张三
    
    这个 U_张三 是张三独有的一切——性格、运气、家庭氛围中
    未被建模的部分。在反事实世界中也保持不变。
  
  Step 2 — Action (干预):
    do(Education = 12)
    切断指向 Education 的箭头，强制设为 12。
    其他方程完全不变。
  
  Step 3 — Prediction (预测):
    仍用同一个方程，同一个 U_张三:
    Income_cf = β · 12 + γ · Ability_张三 + U_张三
    
    如果 β > 0 (教育对收入有正向因果效应):
      Income_cf < 3万  ✓ 收入下降了

结论: SCM 本身已经包含了「学历作为收入方程的加权系数」。
      反事实只需要换一个输入值，不需要另一套模型。
```

**为什么很多人会误以为需要两套模型？**

```
直觉陷阱:
  "大学生的收入形成机制 和 非大学生的收入形成机制 可以不同吗？"
  
  如果机制真的不同 (例如：大学生靠学历溢价，非大学生靠工龄积累)，
  那么反事实推理的前提就崩塌了。

  这恰恰是 Pearl 因果阶梯的精髓:
    结构方程必须描述「变量产生的机制」，
    而非「特定人群的统计模式」。
```

**结构不变性 (Structural Invariance) — 反事实推理的核心前提：**

```
┌──────────────────────────────────────────────────────────┐
│  结构方程 f(Y | parents(Y), U_Y) 必须在干预下保持不变。   │
│                                                          │
│  如果 Education=12 和 Education=16 时 f_Income 本身改变了，│
│  那么用一套方程做反事实推理就失效了。                      │
└──────────────────────────────────────────────────────────┘

失效案例:
  真实世界:  高中生收入 = 工龄 × 0.5  
            大学生收入 = 学历 × 1.0 + 工龄 × 0.3
  
  如果你用「大学生的收入方程」去反事实推「如果没上大学」，
  会得到错误答案 —— 因为方程本身就是错的。
  高中生的收入根本不依赖于「学历」这个变量。

应对策略:

  1. 引入中介变量 — 把机制变化显式建模:
     在 SCM 中添加「行业选择」「职业类型」等中介节点，
     让它们随 Education 变化，而不是让 f_Income 本身变化。
     
         Education → Career_Type → Income
                    ↘────→────────↗
     
     这样 f_Income 保持稳定，变化通过 Career_Type 传导。

  2. 使用非线性方程:
     f 可以包含二次项 (Education²) 或交互项 (Education × Experience)
     自动覆盖不同取值区间的行为差异。

  3. 敏感性分析验证:
     问「如果 f 在干预下偏了 20%，结论还成立吗？」
     如果成立 → 结论稳健
     如果不成立 → 需要更强证据或重新设计模型
```

**数学形式化：**

```
设 SCM M = ⟨U, V, F, P(u)⟩，其中:
  U = 外生变量 (不可观测的个体差异)
  V = 内生变量 (Education, Income, ...)
  F = 结构方程 (每个 V_i = f_i(parents(V_i), U_i))
  P(u) = 外生变量的分布

干预 do(X=x):
  产生子模型 M_x = ⟨U, V, F_x, P(u)⟩
  其中 F_x 是将 F 中的 f_X 替换为常数 x，其余 f_i 不变。

反事实 Y_x(u):
  在子模型 M_x 中，用特定的 u (从观测反推) 计算 Y 的值。
  与观测模型 M 中的 Y 的区别仅在于 f_X 被替换。

关键:
  - 只有一个 F，不是两套 F。
  - M_x 只是 M 的一个「副本」，只改了 f_X 这一行。
  - 反事实 Y_x(u) 和 观测 Y 共享所有 U_i。
```

---

## 八、因果中介分析

### 8.1 中介问题

```
问题: X 对 Y 的效应中, 多少是直接的, 多少是通过中介 M 的？

    X ──→ M ──→ Y
     ↘────────↗

  直接效应 (NDE): X → Y (不经过 M)
  间接效应 (NIE): X → M → Y (通过 M)
  总效应 (TE):    NDE + NIE
```

### 8.2 三种方法

| 方法 | 原理 | 使用场景 |
|------|------|---------|
| **线性路径系数** | NDE = β(X→Y), NIE = β(X→M)×β(M→Y) | 线性 SCM |
| **Baron-Kenny** | 4 步回归: Y~X, M~X, Y~X+M, bootstrap SE | 观测数据 |
| **Pearl 反事实** | E[Y(1,M(0))] - E[Y(0)] 的 Monte Carlo 估计 | 有 SCM 时最精确 |

### 8.3 关键公式

```
Pearl 中介公式:

  NDE = E[Y_{x=1, M=M_{x=0}}] - E[Y_{x=0}]
  NIE = E[Y_{x=1, M=M_{x=1}}] - E[Y_{x=1, M=M_{x=0}}]
  TE  = E[Y_{x=1}] - E[Y_{x=0}] = NDE + NIE

直觉:
  NDE: 给这个人"处理", 但中介保持在"无处理"水平
  NIE: 给这个人"处理", 中介从"无处理"变到"有处理"水平
```

---

## 九、敏感性分析

### 9.1 为什么需要

```
问题: "我们的 ATE 估计假设没有未测量混淆。
      如果这个假设被违反了呢？"

敏感性分析回答:
  "需要多强的未测量混淆才能推翻我们的结论？"
```

### 9.2 Rosenbaum Bounds

```
方法: 假设存在隐藏偏差 Γ, 逐步增大 Γ 直到结论改变

解读:
  "即使存在使 odds 相差 5 倍的未测量混淆,
   我们的结论仍然成立。"
  → 结论非常稳健

  "只需要 Γ=1.1 就能推翻结论"
  → 结论很脆弱, 需要更多数据或更强的设计
```

### 9.3 E-value

```
定义:
  推翻观察到的 ATE 所需的未测量混淆的最小强度,
  用风险比 (Risk Ratio) 表示。

公式: E = RR + √(RR×(RR−1))
      其中 RR = exp(ATE/σ)

解读:
  E = 1.5  → 脆弱   (一个弱混淆就能推翻)
  E = 3.0  → 中等
  E = 8.0  → 稳健   (需要非常强的混淆)
```

---

## 十、物理规律约束

### 10.1 四层约束

```
Layer 1 — DAG 边约束:    物理定律禁止某些因果方向
  e.g. "加速度不能导致力" (违反 F=ma)
  
Layer 2 — SCM 方程替换:  用精确物理公式替代学习到的方程
  e.g. a = F/m 替换回归得到的 a = 0.3*F + 0.01*m
  
Layer 3 — 守恒律验证:    反事实结果必须满足守恒律
  e.g. 动量守恒: Σp_before = Σp_after
  
Layer 4 — 变分原理:      整条轨迹必须满足 δS=0 (最小作用量)
  e.g. 单摆轨迹 θ(t) 必须满足 S = ∫ L dt 取极值
```

### 10.2 15 条物理定律

```
力学 (8): Newton F=ma, Hooke F=−kx, 万有引力, 动能, 动量,
         动量守恒, 能量守恒, 单摆周期, 最小作用量
电磁 (3): Ohm V=IR, Coulomb, Lorentz
热力 (2): 理想气体 PV=nRT, 热传导
流体 (2): Bernoulli, Darcy
```

### 10.3 最小作用量原理 (v0.9.2)

```
作用量: S[q] = ∫ L(q, q̇, t) dt
         L = T - V (动能 - 势能)

变分原理: 真实物理路径使 S 取极值 (δS = 0)
         → Euler-Lagrange 方程: d/dt(∂L/∂q̇) = ∂L/∂q

对因果推断的价值:
  - 验证整条轨迹 (不只是单个点) 是否物理可能
  - 从任意初始路径通过梯度下降恢复稳态路径

系统示例:
  单摆:     L = ½ml²θ̇² - mgl(1-cosθ)  →  θ̈ + (g/l)sinθ = 0
  谐振子:   L = ½mẋ² - ½kx²             →  ẍ + (k/m)x = 0
```

---

## 十一、LLM 自然语言接口

### 11.1 5 步自动管线

```
用户: "吸烟会导致肺癌吗，有哪些混杂因素"

Step 1 — 因果图提取 (LLM):
  DeepSeek API 将自然语言转为结构化 JSON:
  {
    "variables": ["Smoking","LungCancer","Genetics","Age",...],
    "edges": [["Smoking","LungCancer"],["Genetics","Smoking"],...],
    "treatment": "Smoking",
    "outcome": "LungCancer",
    "confounders": ["Genetics","Age"]
  }

Step 2 — 构建 DAG:
  CausalDAG(variables, edges) → 有向无环图

Step 3 — 识别:
  identify_effect(dag, treatment, outcome)
  → back-door / front-door / do-calculus

Step 4 — 估计 (自动):
  a) 无数据？→ 从 DAG 构建线性 SCM, 采样 500 条
  b) 诊断: 残差正态性 ✓ / overlap ✓
  c) 自动选择: linear
  d) ATE=1.54, SE=0.09, CI[1.36,1.72]

Step 5 — LLM 中文解读:
  "吸烟确实会显著增加患肺癌的风险。
   在控制了年龄和遗传因素后,
   吸烟者患肺癌的风险是非吸烟者的约 1.5 倍……"
```

### 11.2 两种运行模式

```
LLM 模式 (推荐):
  > ask 吸烟会导致肺癌吗
  → 5 步全自动, 零人工干预

无 LLM 模式 (离线/精确控制):
  > load simpson              # 加载预置模板
  > effect D R                # 手动跑分析
  > sensitivity D R
  
  > discover data.csv pc      # 从 CSV 发现因果结构
  > effect X Y linear
```

---

## 十二、数据生成方法

### 12.1 线性高斯 SCM 数据生成

```python
from core.discovery import generate_linear_data

# 从已定义的 DAG 生成数据
dag = CausalDAG(["Z","X","Y"], [("Z","X"),("Z","Y"),("X","Y")])
data = generate_linear_data(dag, n_samples=2000, seed=42)
# → np.ndarray shape (2000, 3)

# 内部原理:
# 1. 按拓扑序生成变量
# 2. 每个变量 = Σ(coeff × parent) + noise
# 3. 系数随机采样自 uniform(0.3, 2.0)
# 4. 噪声 = N(0, 1) + random scaling
```

### 12.2 训练数据集生成

```
Type 1 — 结构学习 (100 graphs):
  生成随机 DAG → 采样 500 条 → 保存 CSV + meta JSON
  目标: 训练 Agent 从数据中恢复因果结构

Type 2 — 效应估计 (300 problems):
  已知 DAG + 数据 → 估计 ATE → 与 ground truth 比较
  目标: 评估估计器的准确性

Type 3 — 干预推理 (150 problems):
  给定 DAG + 观测数据 + do(X=x) → 预测干预后 Y 的分布
  目标: 验证 do-calculus 的正确性

Type 4 — 反事实 (96 triplets):
  观测 + 干预 + ground truth 反事实值
  目标: 验证三步反事实的精度

Type 5 — 领域迁移 (8 domains):
  同一因果结构在不同领域的数据 (医学、教育、经济…)
  目标: 训练 Agent 跨领域泛化

总计: 955 个文件, ~15.5 MB
```

### 12.3 自动数据生成 (v0.9.6)

```
Agent 在 LLM 模式下无数据时的自动处理:

1. 从 LLM 提取的 DAG 结构构建线性 SCM
2. 为每条边随机分配系数 (uniform 0.5~2.0)
3. 为每个变量设噪声标准差 0.3
4. 按拓扑序采样 500 条

结果: Agent 可以独立完成完整的因果分析,
      即使没有任何真实数据。
      
限制: 自动生成的数据基于 "线性高斯" 假设,
      不反映真实世界的复杂非线性关系。
      用于快速探索和假设验证,
      不替代真实数据驱动的因果推断。
```

---

## 十三、运行流程

### 13.1 LLM 模式完整流程

```bash
# 1. 配置 API Key (首次, 一次)
echo '{"DEEPSEEK_API_KEY":"sk-xxx"}' > ~/.hermes/causal_config.json

# 2. 启动
cd /home/duyw/causal_agent
python agent.py

# 3. 提问
> ask 教育是否能提高收入？控制家庭背景和能力

# Agent 自动输出:
#   [因果图] → [识别] → [诊断] → [估计] → [中文解读]
```

### 13.2 无 LLM 模式完整流程

```bash
# 1. 加载场景
> load simpson

# 2. 查看因果图
> dag show

# 3. 发现因果结构 (从数据)
> load_data /path/to/data.csv
> discover /path/to/data.csv pc

# 4. 估计效应
> effect D R          # 自动选方法
> effect D R dr       # 指定方法

# 5. 敏感性分析
> sensitivity D R

# 6. 反事实
> whatif D=0 R given G=1,D=1,R=1
```

### 13.3 数据格式

```csv
Gender,Drug,Recovery,Age
F,1,1,45
M,1,1,32
F,0,1,58
M,0,0,41
...
```

要求: 首行表头, 逗号分隔, 数值型, ≥300 行。

---

## 附录 A: 符号约定

| 符号 | 含义 |
|------|------|
| G = (V, E) | 因果图, V=节点, E=有向边 |
| P(y\|do(x)) | 干预分布 |
| ATE | 平均因果效应 |
| CI | 置信区间 |
| SMD | 标准化均值差 |
| NDE / NIE / CDE | 自然直接效应 / 自然间接效应 / 控制直接效应 |
| S = ∫L dt | 作用量泛函 |
| G_{X̄} | 删除 X 出边的图 (do-calculus) |

## 附录 B: 相关文献

| 文献 | 关联 |
|------|------|
| Pearl (2009). Causality (2nd ed.) | do-calculus、SCM、反事实 |
| Spirtes, Glymour & Scheines (2000). Causation, Prediction, and Search | PC、FCI 算法 |
| Chickering (2002). Optimal Structure Identification With Greedy Search | GES 算法 |
| Zhang (2008). On the completeness of orientation rules for FCI | FCI R1-R7 |
| Baron & Kenny (1986). The moderator-mediator variable distinction | 中介分析 |
| Rosenbaum (2002). Observational Studies | 敏感性分析 |
| VanderWeele & Ding (2017). Sensitivity analysis in observational research | E-value |
| LeCun (2022). A Path Towards Autonomous Machine Intelligence | JEPA 架构 |

# Causal Agent — 设计报告

> 版本: v0.9.1 | 日期: 2026-05-14
> 类型: 系统设计白皮书

---

## 摘要

Causal Agent 是一个以**结构因果模型 (SCM)** 为理论内核、以 **do-calculus** 为推理引擎的因果推断智能体。它不依赖大规模训练数据来「学习」因果关系，而是通过**显式的因果图 ($G$) + 结构方程 ($\mathcal{F}$)** 来回答三类核心问题：(1) 观测推理 — 从数据中估计联合分布；(2) 干预推理 — 如果做了 X 会怎样； (3) 反事实推理 — 如果当初没做 X 会怎样。

本文档阐述智能体的设计哲学、系统架构和基本原理。

---

## 一、设计哲学

### 1.1 因果 ≠ 相关

传统机器学习的核心范式是：

$$\text{数据} \xrightarrow{\text{训练}} \text{模型} \xrightarrow{\text{推断}} \text{预测}$$

这个范式在 i.i.d. 假设下有效，但无法回答干预和反事实问题。原因在于：**观测分布 $P(Y \mid X)$ 不等于干预分布 $P(Y \mid do(X))$**。两者的差异来自混杂因子 (confounder)——同时影响 $X$ 和 $Y$ 的第三个变量。

Causal Agent 的设计哲学是：

$$\text{因果图} + \text{数据} \xrightarrow{\text{do-calculus}} \text{识别} \xrightarrow{\text{估计}} \text{效应}$$

关键区别：**结构（因果图）与参数（效应值）分离**。先确定「谁影响谁」，再讨论「影响多大」。

### 1.2 第一性原理

智能体遵循三个第一性原理：

1. **因果优先于相关**：任何统计分析之前，必须先建立因果图。没有因果图的调整可能是无效甚至有害的。

2. **可识别性先于估计**：不要盲目套用回归——先用 do-calculus 判断能否从观测数据中识别因果效应，再选择正确的调整策略。

3. **物理先于拟合**：当已知物理定律（如 $F=ma$、$T=2\pi\sqrt{L/g}$）时，直接使用物理公式替代数据拟合的结构方程。物理定律是不可违背的硬约束。

### 1.3 为什么不需要海量训练数据

因果推断智能体与 LLM/深度学习有本质区别：

| 维度 | 深度学习 | Causal Agent |
|------|---------|-------------|
| 知识来源 | 数据中的统计模式 | 因果图 + 数据 |
| 泛化方式 | i.i.d. 插值 | do-calculus 外推到干预分布 |
| 所需数据量 | 百万级 | 百到千级 |
| 可解释性 | 低（黑盒） | 高（显式 DAG + 方程） |
| 干预推理 | 不支持 | 核心能力 |
| 反事实推理 | 不支持 | 核心能力 |

智能体的「知识」来自两个地方：因果图（由领域专家或因果发现算法提供）和结构方程（由数据估计或物理定律提供）。这两个组件都不需要海量数据。

---

## 二、理论基石

### 2.1 结构因果模型 (SCM)

智能体的理论基础是 Pearl 的结构因果模型。一个 SCM 定义为四元组：

$$\mathcal{M} = \langle U, V, \mathcal{F}, P(u) \rangle$$

其中：
- $U$：外生变量（模型外的原因 + 随机噪声）
- $V$：内生变量（模型内决定的变量）
- $\mathcal{F}$：结构方程集合，$v_i = f_i(\text{PA}(v_i), u_i)$
- $P(u)$：外生变量的联合分布

SCM 的威力在于它统一了三种推理模式：

| 推理模式 | 符号 | 操作 |
|---------|------|------|
| 观测 | $P(V)$ | 从 $P(u)$ 采样，通过 $\mathcal{F}$ 传播 |
| 干预 | $P(V \mid do(X=x))$ | 替换 $f_X$ 为常数 $x$，重算下游 |
| 反事实 | $Y_{X=x}(u) \mid V=v$ | 溯因($u$) → 行动($do$) → 预测($Y$) |

### 2.2 do-calculus

do-calculus (Pearl, 1995) 是一套将干预概率转化为观测概率的规则系统。三条规则对应图上的三种操作：

**Rule 1** — 观测的增删:
$$P(y \mid do(x), z, w) = P(y \mid do(x), w) \quad \text{if} \quad (Y \perp\!\!\!\perp Z \mid X, W)_{G_{\overline{X}}}$$

**Rule 2** — 干预与观测互换:
$$P(y \mid do(x), do(z), w) = P(y \mid do(x), z, w) \quad \text{if} \quad (Y \perp\!\!\!\perp Z \mid X, W)_{G_{\overline{X}\underline{Z}}}$$

**Rule 3** — 干预的增删:
$$P(y \mid do(x), do(z), w) = P(y \mid do(x), w) \quad \text{if} \quad (Y \perp\!\!\!\perp Z \mid X, W)_{G_{\overline{X}, \overline{Z(W)}}}$$

智能体实现了 back-door 调整、front-door 调整和工具变量三种最常见的识别策略，它们是 do-calculus 的特殊情况。

### 2.3 因果发现

当因果图未知时，智能体使用 **PC 算法** (Spirtes et al., 2000) 从纯观测数据中恢复因果结构。算法基于一个核心洞察：数据中的条件独立性关系编码了因果图的结构信息。

$$X \perp\!\!\!\perp Y \mid Z \text{ in data} \iff X \perp\!\!\!\perp_G Y \mid Z \text{ in graph}$$

PC 算法通过系统地检验条件独立性来逐步修剪完全连通图，最终输出 Markov 等价类。

对于存在隐变量的场景，智能体使用 **FCI 算法**，输出 PAG (Partial Ancestral Graph) 并标注可能由隐变量引起的边（$A \leftrightarrow B$）。

---

## 三、系统架构

### 3.1 分层架构

智能体采用四层架构，自底向上依次为：

```
┌──────────────────────────────────────────────────┐
│            Agent Layer (agent.py)                │
│  交互式 REPL · 命令解析 · 自然语言理解            │
├──────────────────────────────────────────────────┤
│           Inference Layer                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │Estimation│ │ Modern   │ │ Sensitivity      │ │
│  │(5种估计器)│ │(DML+CATE)│ │(Rosenbaum+E-value)│ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
├──────────────────────────────────────────────────┤
│           Reasoning Layer                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │Identification│  SCM   │ │   Physics        │ │
│  │(do-calculus) │(do+CF) │ │(16 laws,5 domains)│ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
├──────────────────────────────────────────────────┤
│           Foundation Layer                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ CausalDAG│ │Discovery │ │  Visualization   │ │
│  │(d-sep)   │ │(PC,FCI)  │ │(ASCII,DOT,PNG)   │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
└──────────────────────────────────────────────────┘
```

**Foundation Layer** 提供因果图的基础数据结构和图论算法（d-separation, 道德图检验）。

**Reasoning Layer** 实现因果推理的核心逻辑：效应识别（判断能不能估计）、结构方程模型（干预和反事实）、物理约束引擎（物理定律作为硬约束）。

**Inference Layer** 在 Reasoning Layer 确定了「能不能估计」和「用什么调整集」之后，提供数值估计能力。包含 9 种估计器，覆盖从经典的线性回归到现代的因果森林。

**Agent Layer** 是面向用户的界面层，支持交互式命令行、自然语言解析和完整的演示套件。

### 3.2 模块依赖图

```
graph.py          ← 零外部依赖 (纯 Python + numpy)
identification.py ← graph.py
scm.py            ← graph.py + numpy
discovery.py      ← graph.py + numpy
estimation.py     ← graph.py + numpy
modern.py         ← estimation.py + numpy
sensitivity.py    ← numpy
visualization.py  ← graph.py
physics.py        ← graph.py + scm.py + discovery.py + identification.py
parser.py         ← graph.py + scm.py
agent.py          ← 以上全部
```

### 3.3 数据流

```
用户输入
  │
  ├─ 自然语言 ──→ CausalParser ──→ CausalDAG
  │
  ├─ 数据文件 ──→ PC/FCI/GES ──→ CausalDAG
  │                                    │
  │                              ┌─────┴─────┐
  │                              │ Physics    │
  │                              │ Library    │
  │                              │ 约束+替换  │
  │                              └─────┬─────┘
  │                                    │
  │                         PhysicsInformedCausalGraph
  │                                    │
  │                              ┌─────┴─────┐
  │                              │Identification│
  │                              │ do-calculus  │
  │                              └─────┬─────┘
  │                                    │
  │                          ┌─────────┴─────────┐
  │                          │                   │
  │                    估计阶段              反事实阶段
  │                    ATE / CATE          Abduction→Action
  │                    9 种估计器           →Prediction
  │                          │                   │
  │                          └─────────┬─────────┘
  │                                    │
  │                              敏感性分析
  │                          Rosenbaum + E-value
  │                                    │
  └────────────────────────────────────┘
                                    │
                              自然语言回答
```

---

## 四、关键设计决策

### 4.1 为什么用算法而非神经网络

因果推断的核心挑战不在于函数逼近（神经网络擅长的），而在于**结构识别**（从观测分布中区分 $P(Y \mid X)$ 和 $P(Y \mid do(X))$）。

do-calculus 提供了完整、可证明的识别理论。将这套理论实现为算法，比训练神经网络去模拟它更可靠——因为算法保证正确性，而神经网络只能近似。

智能体在需要函数逼近的地方（如倾向得分估计、结果回归）使用了 ML 模型（DML、因果森林），但在结构推理层面坚持使用显式算法。

### 4.2 为什么物理定律作为硬约束

物理定律与因果推断有天然的亲和性：

- 物理定律天然是因果的：$F=ma$ 意味着力导致加速度，而非加速度导致力
- 物理定律是确定性的：$a=F/m$ 比任何从数据中学到的 $a = \beta_1 F + \beta_2 m + \varepsilon$ 更精确
- 物理定律提供可重复性：同一个公式在火星上同样有效，而数据拟合的模型只能在训练分布内有效

因此，智能体将物理定律设计为**硬约束**——一旦匹配到已知物理定律，直接替换结构方程。这与纯数据驱动的方法互补：物理定律处理已知部分，数据拟合处理未知部分。

### 4.3 为什么五类训练数据而非一个大数据集

智能体需要验证的能力不是单一任务，而是五个维度：

| 类型 | 验证的能力 |
|------|-----------|
| Type 1: 结构学习 | 能否从纯数据中恢复因果图 |
| Type 2: 效应估计 | 给定 DAG，能否正确估计 ATE |
| Type 3: 干预推理 | 能否区分 $P(Y \mid T)$ 和 $P(Y \mid do(T))$ |
| Type 4: 反事实 | 个体层级的预测是否正确 |
| Type 5: 领域迁移 | 同构因果结构在跨域时是否一致 |

这五类数据集的共同目标是：验证智能体是否真正掌握了**因果推理的抽象能力**，而非只是记住了某个特定领域的模式。

---

## 五、能力边界

### 5.1 当前能力

| 能力 | 状态 | 验证结果 |
|------|:---:|---------|
| 给定 DAG + 数据，估计 ATE | ✅ | 偏差消除 90% |
| 跨领域一致推理 | ✅ | 8 领域 ATE=0.6±0.02 |
| 线性 SCM 反事实 | ✅ | C=2.200 (精确匹配) |
| 从数据自动发现因果结构 | ✅ | PC 完美恢复 Simpson DAG |
| 处理隐变量 (FCI) | ✅ | PAG 输出 |
| 异质性效应 (CATE) | ✅ | 4 种 meta-learner + 因果森林 |
| 敏感性分析 | ✅ | Rosenbaum + E-value |
| 物理定律自动匹配 | ✅ | 摆钟 $T=2\pi\sqrt{L/g}$ |
| 9 种 ATE 估计器 | ✅ | 覆盖线性到非参数 |

### 5.2 当前局限

| 局限 | 影响 | 缓解方案 |
|------|------|---------|
| PC/FCI 对噪声敏感 | 因果发现可能不稳定 | 自举置信度 |
| 线性 SCM 假设 | 反事实在非线性场景可能不准 | Phase 3 DML 部分缓解 |
| 无 LLM 集成 | 需要手动指定因果图 | Phase 4 LLM 原型就绪 |
| 仅支持表格数据 | 无法处理图像/文本 | 需要因果表示学习 (Phase 6) |

---

## 六、演进路线

智能体遵循六阶段渐进式演进路线，当前处于 Phase 4：

```
Phase 1 ✅ 补全推理管线  ████████████████░░░░  40%
  → 效应估计 + 反事实修复 + 敏感性 + 可视化

Phase 2 ✅ 高级因果发现  ████████████████████  55%
  → FCI (隐变量) + 自举置信度

Phase 3 ✅ 现代方法      ████████████████████  70%
  → DML + CATE (S/T/X/Forest) + do-why

Phase 4 🔧 LLM + 物理    ████████████████████  85%
  → LLM 原型 + 物理因果引擎 (16 条定律)

Phase 5 ⏳ 产品化                        (待开始)
  → Web UI + API Server + Docker 部署

Phase 6 ⏳ 前沿探索                      (持续)
  → 时间序列因果发现 + 因果表示学习 + 因果 RL
```

---

## 附录: 项目文件索引

### 核心引擎 (10 个模块)
| 文件 | 行数 | 功能 |
|------|:---:|------|
| `core/graph.py` | ~300 | CausalDAG, d-separation |
| `core/identification.py` | ~470 | back-door, front-door, IV, do-calculus |
| `core/scm.py` | ~390 | SCM, do(), counterfactual |
| `core/discovery.py` | ~1100 | PC, FCI, GES, bootstrap |
| `core/estimation.py` | ~720 | 5 种 ATE 估计器 |
| `core/modern.py` | ~600 | DML, CATE (S/T/X/Forest) |
| `core/sensitivity.py` | ~230 | Rosenbaum, E-value |
| `core/visualization.py` | ~200 | ASCII, DOT, Mermaid, PNG |
| `core/physics.py` | ~900 | 物理因果引擎, 14 条定律 |
| `agent.py` | ~560 | 交互式命令行 |

### 演示 (4 个)
| 文件 | 功能 |
|------|------|
| `demos/run_all.py` | 全功能验证 (5 个测试) |
| `demos/llm_prototype.py` | LLM 集成原型 |
| `demos/physics_causal_demo.py` | 物理因果集成演示 |
| `demos/aircraft_disappearance.py` | 飞机消失预测 |

### 文档 (8 份)
| 文件 | 内容 |
|------|------|
| `ARCHITECTURE.md` | 系统架构 |
| `MODELS_AND_ALGORITHMS.md` | 模型与算法 (LaTeX, 875 行) |
| `DESIGN_REPORT.md` | 本文档 — 设计报告 |
| `PHYSICS_EXTENSION_GUIDE.md` | 物理规律扩展指南 |
| `ROADMAP.md` | 六阶段路线图 |
| `RUNNING.md` | Linux 运行指南 |
| `DATA_REQUIREMENTS.md` | 数据需求方案 |

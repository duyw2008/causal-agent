# Causal Agent — 系统架构文档

> 最后更新: 2026-05-14
> 状态: v0.9.6 — Phase 4 完成 + LLM 集成 (DeepSeek) + 自主诊断与自动方法选择
> 测试: 52/52 passing

---

## 一、设计理念

**结构与参数分离。** 先通过因果图确定"谁影响谁"，再讨论"影响多大"。

```
用户输入 (自然语言 / 数据文件 / 交互命令)
        │
        ▼
   ┌─────────────┐
   │  NL Parser  │  "X causes Y" → edge X→Y
   │  Discovery  │  data.csv → PC/GES → CausalDAG
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  CausalDAG  │  有向无环图 + d-separation
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │Identification│  do-calculus: back-door / front-door / IV
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │    SCM      │  结构方程 → 干预 → 反事实
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │  回答引擎   │  可识别？调整集？效应值？反事实？
   └─────────────┘
```

---

## 二、模块架构

```
causal_agent/
│
├── agent.py                    [✅] 交互式 REPL + 演示
│
├── core/
│   ├── graph.py                [✅] CausalDAG + d-separation
│   ├── identification.py       [✅] back-door / front-door / IV / do-calculus
│   ├── scm.py                  [✅] SCM + do()干预 + 反事实
│   ├── discovery.py            [✅] PC / FCI / GES 算法
│   ├── estimation.py           [✅] 五类 ATE 估计器 (Phase 1)
│   ├── sensitivity.py          [✅] Rosenbaum + E-value (Phase 1)
│   ├── visualization.py        [✅] ASCII/DOT/Mermaid/PNG (Phase 1)
│   ├── modern.py               [✅] DML + CATE + do-why (Phase 3)
│   ├── physics.py              [✅] 物理因果引擎 (Phase 4)
│   ├── ts_discovery.py         [✅] 时间序列因果发现 (Phase 4)
│   ├── mediation.py            [✅] 因果中介分析 NDE/NIE/CDE (v0.9.3)
│   └── llm_client.py           [✅] DeepSeek LLM 客户端 (v0.9.5)
│
├── nlp/
│   └── parser.py               [✅] NL → CausalDAG (规则 + 模板)
│
├── datasets/                   [✅] 五类因果推理训练数据
│   ├── README.md               # 数据集文档 (schema, 使用说明)
│   ├── type1_structure_learning/  # 100 个随机 DAG × 500 样本
│   ├── type2_effect_estimation/   # 300 个 ATE 估计问题 × 1000 样本
│   ├── type3_interventional/      # 150 个干预推理问题
│   ├── type4_counterfactual/      # 96 个反事实三元组
│   └── type5_domain_transfer/     # 8 领域跨域迁移数据
│
├── training_data.py            [✅] 五类训练数据生成器
├── generate_all_datasets.py    [✅] 批量生成 + 存储脚本
│
├── demos/                      [✅] 功能演示与原型验证
│   ├── run_all.py              # 5 个核心 demo（全流程验证）
│   ├── llm_prototype.py        # LLM 集成原型（Phase 4 预览）
│   └── physics_causal_demo.py # 物理规律 + 因果推断集成原型
│
├── ARCHITECTURE.md             [✅] 系统架构
├── LEARNING.md                  [✅] 学习文档 — 从零理解全部算法与运行原理
├── MODELS_AND_ALGORITHMS.md    [✅] 模型与算法设计 (LaTeX 数学)
├── CAPABILITIES.md              [✅] 能力全景图 — 全功能速览
├── JEPA_CAUSAL_ARCHITECTURE.md  [✅] JEPA × Causal 合成架构分析
├── PHYSICS_EXTENSION_GUIDE.md  [✅] 物理规律扩展指南
├── PHYSICS_LEAST_ACTION.md    [✅] 最小作用量原理集成方案
├── DATA_REQUIREMENTS.md        [✅] 数据需求与训练方案
├── ROADMAP.md                  [✅] 六阶段完善路线图
├── CHANGELOG.md                [✅] 变更日志
└── RUNNING.md                  [✅] Linux 运行指南
```

### 3.7

## 三、各文件功能详解

### 3.1 `core/graph.py` — 因果 DAG + d-separation [✅]

**CausalDAG** — 因果图的基础数据结构。

```
CausalDAG(variables, edges)
│
├── parents(v) / children(v)          # 直接因果邻居
├── ancestors(v) / descendants(v)     # 传递闭包
├── topological_order()               # 因果序 (用于 SCM 采样)
│
├── is_d_separated(X, Y, Z)           # d-separation 检验
│   └── 算法: 道德图法 (Moral Graph)
│       1. 取 X∪Y∪Z 的祖先子图
│       2. 道德化: 连接父节点对, 变无向图
│       3. 删除 Z 节点及边
│       4. X 与 Y 不连通 → d-separated
│       复杂度: O(|V| + |E|)
│
├── mutilate(X)                       # do(X) 后的子图 (删除入边)
├── remove_outgoing(X)                # 删除出边 (do-calculus 用)
│
└── to_mermaid() / summary()          # 可视化
```

### 3.2 `core/identification.py` — 因果效应可识别性 [✅]

**identify_effect(dag, treatment, outcome) → IdentificationResult**

```
IdentificationResult
├── identifiable: bool       # 能否从观测数据估计？
├── method: str              # back-door / front-door / IV / d-separation
├── adjustment_set: [str]    # 需要调整的变量集
└── expression: str          # 可估计的表达式
```

三种识别策略:

| 策略 | 条件 | 公式 |
|------|------|------|
| Back-door | 存在 Z 阻断所有后门路径 | P(y\|do(x)) = Σ_z P(y\|x,z)P(z) |
| Front-door | 存在中介 M: X→M→Y 且满足阻断条件 | P(y\|do(x)) = Σ_m P(m\|x) Σ_x' P(y\|x',m)P(x') |
| IV | 工具变量 Z 满足三条件 | ATE = Cov(Y,Z)/Cov(X,Z) |

辅助函数:
- `find_back_door_adjustment()` — 最小后门调整集
- `find_front_door_adjustment()` — 前门中介变量
- `check_instrument()` — IV 有效性检验
- `descendants_union()` — 后代集合的并集
- `do_calculus_rule1/2/3()` — Pearl 的三条规则

### 3.3 `core/scm.py` — 结构因果模型 [✅]

**SCM(dag, equations)** — 在 DAG 上附加结构方程。

```
SCM
├── sample(n)                       # 从联合分布采样
├── intervene({X: x}) → IntervenedSCM  # do(X=x)
│   └── IntervenedSCM.sample(n)     # 干预后分布采样
│
└── counterfactual(obs, intv, target)  # 三步反事实
    ├── Abduction: 从观测反推外生噪声 u
    ├── Action: 施加干预 do(X=x)
    └── Prediction: 计算目标变量新值

StructuralEquation(variable, func, parents, noise_dist)
│
└── evaluate(parent_values, noise) → 计算变量值

工厂函数:
├── linear_eq(coeffs, intercept)    # 创建线性方程
└── linear_scm(dag, coefficients)   # 快速创建线性 SCM
```

### 3.4 `core/discovery.py` — 因果发现 [✅]

**从纯观测数据中学习因果结构。**

```
条件独立性检验:
├── fisher_z_test(data, x, y, cond, alpha)
│   └── 连续变量: 偏相关 + Fisher z-transform
│       偏相关 = corr(resid_X, resid_Y)  其中残差 = 回归掉 Z 后的剩余
│       z = 0.5·ln((1+r)/(1-r))·√(n-k-3)
│
└── g_squared_test(data, x, y, cond, alpha)
    └── 离散变量: G² 似然比检验 → χ² 近似

PC 算法:
├── 输入: data[n_samples × n_vars]
├── 输出: CausalDAG (CPDAG 的 DAG 表示)
│
├── Step 1: 骨架学习
│   └── 从完全图开始, k=0,1,2,... 逐步增大条件集
│       如果 X ⊥ Y | S (大小为 k) → 删除边 X—Y
│
├── Step 2: v-structure 定向
│   └── 对 X—Z—Y (X 与 Y 不相邻):
│       如果 Z ∉ SepSet(X,Y) → X→Z←Y
│
└── Step 3: Meek 规则传播
    ├── R1: a→b—c 且 a,c 不相邻 → b→c
    ├── R2: a→b→c 且 a—c → a→c
    └── R3: a—b→c, a—d→c, b,d 不相邻, a—c → a→c

GES 算法:
├── 输入: data, 评分函数 = BIC
├── 输出: CausalDAG
│
├── Phase 1 (Forward): 不断加边, 选 BIC 提升最大的
├── Phase 2 (Backward): 不断删边, 选 BIC 提升最大的
└── Phase 3 (CI Pruning): 条件独立性剪枝
    └── 对每条边测试 X ⊥ Y | S (Fisher z)
        如果存在分离集 S → 删除边 X→Y
        消除 collider X→Z←Y 中 BIC 误导的 X→Y 多余边

辅助函数:
└── generate_linear_data(dag, n_samples) → np.ndarray
    根据 DAG 生成合成数据 (用于测试)
```

### 3.5 `nlp/parser.py` — 自然语言解析器 [✅]

**CausalParser(text)** — 将 NL 描述转为因果模型。

```
支持模式:
├── 变量提取: "variables: X, Y, Z" / 大写词 / 因果语句
├── 边提取: "X causes Y" / "X → Y" / "X affects Y"
├── 系数提取: "X = 0.5*Y" / "affects Y by 0.5"
├── 查询提取: "effect of X on Y" / "do(X=1)"
│
├── build_dag() → CausalDAG
└── build_scm() → SCM (需系数)

预置模板 (load_template):
├── simpsons_paradox  — G→D, G→R, D→R
├── smoking_lung_cancer — G→S,C; S→T,C; T→C
├── education_income  — S→E,I; E→I
├── front_door_example — X→M, M→Y (未观测混淆)
└── m_bias            — X→Y (collider Z)
```

### 3.6 `agent.py` — 交互式命令行 [✅]

### 3.7 `training_data.py` — 训练数据生成器 [✅]

### 3.9 `datasets/` — 训练数据集 [✅]

955 个文件, ~15.5 MB。五类数据集:

| 类型 | 数量 | 格式 | 训练目标 |
|------|------|------|---------|
| Type 1 结构学习 | 100 图 | CSV + JSON | 数据 → DAG |
| Type 2 效应估计 | 300 问题 | CSV + JSON | (DAG,数据) → ATE |
| Type 3 干预推理 | 150 问题 | JSON (meta) | (DAG,数据,do) → E[Y\|do] |
| Type 4 反事实 | 96 三元组 | JSONL | (SCM,obs,do) → Y_cf |
| Type 5 领域迁移 | 8 域 | CSV + JSON | 同构因果结构跨域泛化 |

### 3.10 `demos/run_all.py` — 功能验证套件 [✅]

5 个核心 demo，覆盖全功能：
- Simpson's Paradox 全流程（识别→估计→敏感性）
- 因果发现（PC/FCI/Bootstrap）
- 反事实推理（溯因→行动→预测）
- 现代方法（DML + CATE，9 种估计器对比）
- 领域迁移（8 领域 ATE 恢复，偏差消除 97%）

运行: `python demos/run_all.py`

### 3.11 `demos/llm_prototype.py` — LLM 集成原型 [✅]

展示接入 LLM 后的能力提升：
- 自由文本 → 因果图（零手动）
- 技术结果 → 自然语言解读
- 领域知识注入（混杂因子建议）
- 反事实叙事生成

运行: `python demos/llm_prototype.py`

### 3.12 `demos/physics_causal_demo.py` — 物理规律集成 [✅]

物理规律作为因果约束的三种形式：
- DAG 约束（物理禁止某些因果方向，如 $a \to F$）
- SCM 方程替换（用 $a=F/m$ 替代学习的线性方程）
- 守恒律验证（反事实必须满足动量/能量守恒）

运行: `python demos/physics_causal_demo.py`

```
命令:
  load <描述>        # 自然语言加载场景
  template <名称>    # 加载预置模板
  discover <file>    # 从 CSV 数据中学习因果图
  effect <X> <Y>     # 识别因果效应
  whatif X=1.0 Y     # 干预预测
  whatif X=0 Y given X=2,Y=3  # 反事实推断
  explain <概念>     # 解释 backdoor/frontdoor/d-sep 等
  model              # 显示当前 DAG
  demo               # 运行全部演示
```

---

## 四、数据流图

```
场景描述 (NL) ──→ CausalParser ──→ CausalDAG + (可选) SCM
                                         │
数据文件 (.csv) ──→ PC / GES ────────────┘
                                         │
                                    ┌────┴────┐
                                    │         │
                              识别阶段     估计阶段
                              identify_effect()  scm.intervene()
                                    │         │
                                    │    ┌────┴─────┐
                                    │    │ Back-door │
                                    │    │ Front-door│
                                    │    │    IV     │
                                    │    └────┬─────┘
                                    │         │
                                    └────┬────┘
                                         │
                                    回答引擎
                                    │
                              "P(Y|do(X)) 可通过调整
                               {Z1,Z2} 估计。调整集已验证
                               无后门路径。"
```

---

## 五、依赖关系

```
graph.py          ← 零外部依赖
identification.py ← graph.py
scm.py            ← graph.py + numpy
discovery.py      ← graph.py + numpy + scipy
parser.py         ← graph.py + scm.py
agent.py          ← 以上全部
```

外部依赖:
- `numpy`  — SCM 采样, 偏相关计算
- `scipy`  — Fisher z-transform, χ² 检验, 最小二乘

---

## 六、测试覆盖

| 测试 | graph | identification | scm | discovery | estimation | sensitivity | viz | parser | agent | time_series | datasets |
|------|-------|---------------|-----|-----------|------------|-------------|-----|--------|-------|-------------|----------|
| Chain X→Y→Z | ✅ | ✅ | ✅ | ✅ | ✅ | — | — | — | — | — | ✅ |
| Fork Z→X,Z→Y | ✅ | ✅ | — | ✅ | — | — | — | — | — | — | ✅ |
| Collider X→Z←Y | ✅ | ✅ | — | ✅ | — | — | — | — | — | — | ✅ |
| Simpson 混杂 | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ |
| Smoking 4-var | ✅ | ✅ | ✅ | — | — | — | — | ✅ | ✅ | — | — |
| SCM 干预+反事实 | — | — | ✅ | — | — | — | — | — | ✅ | — | — |
| 连续处理 ATE | — | — | — | — | ✅ | ✅ | — | — | — | — | — |
| NL 解析 | — | — | — | — | — | — | — | ✅ | ✅ | — | — |
| 物理因果引擎 | ✅ | ✅ | ✅ | ✅ | — | — | — | — | — | — | — |
| 五类数据生成 | — | — | — | — | — | — | — | — | — | — | ✅ 955文件 |

# Causal Agent 能力全景图

> 版本: v0.9.2 (2026-05-14)
> 路径: `/home/duyw/causal_agent/`
> 测试: 52/52 passing

---

## 一、核心能力矩阵

| 能力域 | 功能 | 支持程度 |
|--------|------|:---:|
| **因果图建模** | DAG 构建、变量管理、边操作、拓扑排序 | ✅ 完整 |
| **结构学习** | 从数据中自动发现因果结构 | ✅ 3 种算法 |
| **因果效应识别** | back-door, front-door, IV, do-calculus | ✅ 4 类策略 |
| **效应估计** | ATE, ATT, CATE, 分层效应 | ✅ 10+ 估计器 |
| **反事实推理** | SCM 三步反事实、非线性 SCM | ✅ 完整 |
| **干预推断** | do() 算子、干预分布采样 | ✅ 完整 |
| **敏感性分析** | Rosenbaum bounds, E-value | ✅ 完整 |
| **物理规律约束** | 14 条物理定律、守恒律验证 | ✅ 完整 |
| **自然语言解析** | 中文/英文自由文本 → DAG + SCM | ✅ 规则+模板 |
| **时间序列因果** | Granger, TS-PC, PCMCI-lite | ✅ 原型 |
| **交互式 CLI** | 15+ 命令、自动补全、彩色输出 | ✅ 完整 |

---

## 二、因果发现算法

### 2.1 PC 算法 (约束方法)

```
输入:   观测数据 + 显著性水平 α
输出:   CausalDAG (CPDAG 等价类)
原理:   条件独立性检验 → 骨架学习 → v-structure 定向 → Meek 规则传播
修复:   Kahn 拓扑排序引导 + 索引回退 → 零环路保证
```

### 2.2 FCI 算法 (隐变量支持)

```
输入:   观测数据 + α
输出:   PAG (Partial Ancestral Graph, 7 种边类型)
原理:   PC 骨架 + 额外 CI 检验 → FCI 定向规则 R1-R7 (Zhang 2008)
边类型: → (directed), ◦→ (ancestral), ◦—◦ (undetermined), ↔ (confounded)
```

### 2.3 GES 算法 (评分方法, 三阶段)

```
Phase 1 (Forward):     空图 → 贪心加边 (BIC 提升最大)
Phase 2 (Backward):    删边 (BIC 提升最大)
Phase 3 (CI Pruning):  条件独立性剪枝 (消除 collider 多余边)
```

### 2.4 自举置信度

```python
conf = bootstrap_edge_confidence(data, var_names, method="pc", n_bootstrap=100)
# 返回: {"X→Y": 0.87, "X→Z": 0.42, ...}
```

---

## 三、因果效应识别

### 3.1 Back-door 调整

```
条件:  ∃Z 阻断 T→Y 的所有后门路径，且 Z 不含 T 的后代
公式:  P(y|do(x)) = Σ_z P(y|x,z) P(z)
```

### 3.2 Front-door 调整

```
条件:  ∃M: X→M→Y, X→M 无后门, M→Y 的后门已通过 X 阻断
公式:  P(y|do(x)) = Σ_m P(m|x) Σ_x' P(y|x',m) P(x')
```

### 3.3 工具变量 (IV)

```
条件:  Z 满足 (1) Z 与 T 相关, (2) Z 仅通过 T 影响 Y, (3) Z 与 Y 之间无不开放的混淆
公式:  ATE = Cov(Y,Z) / Cov(T,Z)
```

### 3.4 do-calculus 三条规则 (Pearl)

```
Rule 1:  P(y|do(x),z,w) = P(y|do(x),w)       if Y ⊥ Z | X,W in G_{X̅}
Rule 2:  P(y|do(x),do(z),w) = P(y|do(x),z,w)  if Y ⊥ Z | X,W in G_{X̅,Z̲}
Rule 3:  P(y|do(x),do(z),w) = P(y|do(x),w)    if Y ⊥ Z | X,W in G_{X̅,Z(W)̅}
```

---

## 四、效应估计器

### 4.1 经典估计器 (core/estimation.py)

| 估计器 | 方法 | 输出 | 调用 |
|--------|------|------|------|
| LinearRegression | OLS 回归 | ATE ± 95% CI | `estimate_effect(..., "linear")` |
| PropensityScoreMatching | 逻辑回归 + 匹配 | ATT, ATE | `estimate_effect(..., "psm")` |
| InverseProbWeighting | 倾向得分倒数加权 | ATE, 权重诊断 | `estimate_effect(..., "ipw")` |
| DoublyRobust | PSM + OLS 结合 | ATE (双重鲁棒) | `estimate_effect(..., "dr")` |
| Stratification | 分层 Mantel-Haenszel | 分层 ATE | `estimate_effect(..., "stratified")` |

### 4.2 现代估计器 (core/modern.py)

| 估计器 | 方法 | 输出 | 调用 |
|--------|------|------|------|
| Double ML | 交叉拟合 + Neyman 正交 | ATE ± CI | `estimate_ate_dml(...)` |
| S-learner | 单一模型 (T 作为特征) | CATE | `estimate_cate_slearner(...)` |
| T-learner | 处理组/对照组分别建模 | CATE | `estimate_cate_tlearner(...)` |
| X-learner | 处理组建模 + 虚拟对照组 | CATE | `estimate_cate_xlearner(...)` |
| CausalForest | 树模型 + 诚实估计 | CATE | `estimate_cate_forest(...)` |
| do-why 后端 | Microsoft do-why 库 | ATE | `estimate_ate_dowhy(...)` |

### 4.3 auto 模式

```python
est = estimate_effect(data, vars, treatment, outcome, adjustment_set, "auto")
# 自动选择最佳估计器: DR > IPW > PSM > Linear > Stratified
```

---

## 五、敏感性分析

### 5.1 Rosenbaum Bounds

```
检验: "需要多大的隐藏偏差 Γ 才能改变结论？"
输出: 临界值 Γ_critical + 置信区间
```

### 5.2 E-value

```
检验: "未观测混淆需要多强才能解释观测效应？"
公式:  E = RR + √(RR×(RR−1))
      其中 RR = exp(ATE/σ)  (对于连续结局)
```

### 5.3 完整报告

```python
report = full_sensitivity_report(est, data, est.variable_names, treatment, outcome)
# 输出: Rosenbaum 临界值 + E-value + 联合解释
```

---

## 六、反事实推理

### 6.1 三步反事实 (Pearl)

```
Step 1 - Abduction:  从观测反推外生噪声 u = observed − f(parents)
Step 2 - Action:     施加干预 do(X=x)，修改结构方程
Step 3 - Prediction:  用修改后的模型 + 外生噪声 u 计算结果
```

### 6.2 非线性 SCM

```python
def nonlinear_eq(coeff_terms: Callable, noise: float = 0.0):
    """
    支持: 二次 Y = a*X² + b*Z + c
         交互 Y = a*X + b*Z + c*X*Z
         Cobb-Douglas Y = A * X^α * Z^β
         Sigmoid  Y = 1/(1 + exp(-(a*X + b)))
    """
```

### 6.3 干预推断

```python
scm.intervene({"X": 1.0})      # do(X=1) → 修改 SCM
intv_scm.sample(1000)           # 从 P(y|do(X=1)) 采样
```

---

## 七、物理规律集成

### 7.1 三层约束

| 层次 | 约束类型 | 机制 |
|------|---------|------|
| DAG 层 | 因果方向约束 | 物理定律禁止的边被标记为 forbidden |
| SCM 层 | 方程替换 | PhysicsInformedSCM 用精确方程替换学习到的边 |
| 守恒律层 | 输出验证 | 反事实结果必须满足能量/动量守恒 |
| **变分层** | 轨迹验证 | 整条演化路径必须满足 δS=0 |

### 7.2 定律库 (15条)

| 领域 | 定律数 | 示例 |
|------|:---:|------|
| 力学 | 8 | Newton F=ma, Hooke F=−kx, 万有引力, **最小作用量** |
| 电磁学 | 3 | Coulomb, Ohm, Lorentz |
| 热力学 | 2 | 理想气体 PV=nRT, 热传导 |
| 流体 | 2 | Bernoulli, 达西定律 |

### 7.3 符号公式发现

```python
SymbolicPhysicsDiscovery:  从数据中自动发现物理公式
  - 符号回归: 用符号组合拟合数据
  - 维度分析: 检查量纲正确性
  - 与已知定律库比对
```

### 7.4 最小作用量原理 (v0.9.2 新增)

```python
# 拉格朗日力学系统
from core.physics import simple_pendulum, harmonic_oscillator, ActionPrinciple

pendulum = simple_pendulum(l=1.0, g=9.81)
principle = ActionPrinciple(pendulum, tolerance=0.02)

# 验证轨迹
result = principle.validate_trajectory(theta_path, dt=0.01)
print(result["valid"])        # True/False
print(result["action"])       # 作用量 S
print(result["max_gradient"]) # max|δS/δq|

# 从任意初始路径寻找稳态路径
opt = principle.find_stationary_path(q_start=0.5, q_end=-0.3, n_steps=100, dt=0.01)
```

---

## 八、交互式 Agent 命令

### 8.1 场景加载

```python
> load simpson           # 加载预置模板
> scenario X causes Y; Y affects Z; Z causes W   # 自由文本解析
```

### 8.2 因果分析

```python
> effect D on R          # 估计因果效应
> whatif D = 0 on R      # 干预推演 (do-calculus)
> identify D R           # 仅识别 (不估计)
```

### 8.3 因果发现

```python
> discover data.csv pc              # PC 算法
> discover data.csv ges --bootstrap=100   # GES + 自举
> discover data.csv fci             # FCI (含隐变量)
```

### 8.4 可视化

```python
> dag show               # ASCII 图
> dag save png /tmp/dag.png   # PNG/SVG/PDF/DOT
```

### 8.5 敏感性

```python
> sensitivity D R        # 敏感性分析报告
```

### 8.6 帮助与学习

```python
> help                    # 所有命令
> explain backdoor        # 概念解释
> demo                    # 运行演示
```

---

## 九、数据处理

### 9.1 训练数据集 (5类, 955文件)

| 类型 | 问题数 | 目标 |
|------|:---:|------|
| Type 1: 结构学习 | 100 | 数据 → DAG |
| Type 2: 效应估计 | 300 | (DAG, 数据) → ATE |
| Type 3: 干预推理 | 150 | (DAG, 数据, do) → E[Y\|do] |
| Type 4: 反事实 | 96 | (SCM, obs, do) → Y_cf |
| Type 5: 领域迁移 | 8 domains | 同构因果结构跨域泛化 |

### 9.2 数据生成

```python
from core.discovery import generate_linear_data
data = generate_linear_data(dag, n_samples=2000, seed=42)  # → np.ndarray
```

---

## 十、当前限制与待完善

| 项目 | 状态 | 优先级 |
|------|------|:---:|
| LLM 真实接入 (API) | 仅原型 `demos/llm_prototype.py` | P1 |
| PAG 原生可视化 | `dag show` 暂用 to_dag() 转换 | P2 |
| Type 3 数据集缺原始 CSV | 有 meta (150个), 无 data 文件 | P2 |
| 因果中介分析 (mediation) | 未实现 | P2 |
| Web UI / REST API | 未实现 (Phase 5) | P2 |
| Docker 部署 | 未实现 (Phase 5) | P2 |
| 因果公平性 (fairness) | 未实现 (Phase 6) | P3 |
| 强化学习 + 因果 (causal bandit) | 未实现 (Phase 6) | P3 |
| 因果表示学习 | 未实现 (Phase 6) | P3 |

---

## 十一、快速开始

```bash
cd /home/duyw/causal_agent

# 交互式
python agent.py

# 运行演示
python demos/run_all.py

# 教程
python demos/tutorial.py

# 物理因果
python demos/physics_causal_demo.py

# 最小作用量原理
python demos/least_action_demo.py

# 运行全部测试 (52个)
python -c "
import sys, os, importlib
sys.path.insert(0,'.')
os.chdir('/home/duyw/causal_agent')
modules = ['tests.test_graph','tests.test_identification','tests.test_scm',
           'tests.test_discovery','tests.test_estimation',
           'tests.test_sensitivity_physics','tests.test_ts_parser']
for m in modules:
    mod = importlib.import_module(m)
    for name in dir(mod):
        obj = getattr(mod,name)
        if isinstance(obj,type) and name.startswith('Test'):
            if hasattr(obj,'setup_class'): obj.setup_class()
            for mn in dir(obj):
                if mn.startswith('test_'):
                    inst=obj()
                    if hasattr(inst,'setUp'): inst.setUp()
                    getattr(inst,mn)()
print('52/52 passing')
"
```

---

## 十二、文档体系

| 文档 | 路径 | 内容 |
|------|------|------|
| 架构 | ARCHITECTURE.md | 系统设计、模块树、文件功能 |
| 模型与算法 | MODELS_AND_ALGORITHMS.md | LaTeX 数学推导 (875行, 405公式) |
| 设计报告 | DESIGN_REPORT.md | 设计哲学、架构原理 |
| 路线图 | ROADMAP.md | 六阶段计划 (当前 v0.9.1, 88%) |
| 运行指南 | RUNNING.md | 环境配置、运行、调试 |
| 变更日志 | CHANGELOG.md | v0.5 → v0.9.1 全历史 |
| 物理扩展 | PHYSICS_EXTENSION_GUIDE.md | 三维扩展流程、代码模板 |
| 能力全景 | CAPABILITIES.md | 本文档 — 全功能速览 |

# Causal Agent — 变更日志 (Changelog)

> 记录所有版本变化、模型修改和程序变更

---

## v0.9.7 (2026-05-18) — 文档同步与质量审计

### 文档修复

| 项目 | 文件 | 说明 |
|------|------|------|
| §10 过时信息 | CAPABILITIES.md | LLM API、mediation、Type 3 数据集 3 项状态已过时 → 同步为完成 |
| 数据集统计 | CAPABILITIES.md | 955→1079 文件，5 类精确计数 (541 CSV + 538 JSON) |
| Phase 5 进度 | ROADMAP.md | 95%→5%（Web UI/API/Docker 均未开始） |
| Phase 6 进度 | ROADMAP.md | 100%→30%（仅部分原型就绪） |
| 总体进度 | ROADMAP.md | 95%→72%（重新加权计算） |
| Demo 数量 | ARCHITECTURE.md, 技能文档 | 统一为"6 个独立文件 + run_all 内含 8 个场景" |
| SyntaxWarning | core/ts_discovery.py | docstring 中 `\ ` 非法转义序列修复 |

### 质量审计结果

| 检查项 | 结果 |
|--------|:--:|
| 测试 (52/52) | ✅ |
| Demo (8/8) | ✅ |
| 核心模块导入 (13/13) | ✅ |
| 文档版本一致性 (14 docs) | ✅ |
| 数据集完整性 (541 CSV + 538 JSON) | ✅ |

### 模块变更

无代码功能变更 — 纯文档同步与质量修复。

---

## v0.9.6 (2026-05-14) — 自主诊断 + 自动数据生成 + 自动方法选择

### 新增能力

| 能力 | 实现位置 | 说明 |
|------|---------|------|
| 自动数据生成 | `agent.py` → `ask_llm()` | 无数据时从因果图自动构建线性 SCM 并采样 500 条 |
| 假设诊断 | `agent.py` → `_run_diagnostics()` | 残差正态性检查 (skewness+kurtosis) + 协变量重叠性 (SMD) |
| 自动方法选择 | `agent.py` → `_auto_select_method()` | linear → PSM(重叠差) → IPW(非线性) → DR(都差) |
| API Key 持久化 | `core/llm_client.py` | 配置文件 `~/.hermes/causal_config.json` 自动读取 |

### 文档

| 文档 | 更新 |
|------|------|
| RUNNING.md | 重写 — LLM 模式 / 无LLM模式双轨指南 + 命令速查 + FAQ |

---

## v0.9.5 (2026-05-14) — DeepSeek LLM 集成

### 新增模块

| 模块 | 文件 | 说明 |
|------|------|------|
| LLM 客户端 | `core/llm_client.py` | DeepSeek API 客户端 (urllib, 零依赖) |
| 自然语言接口 | `agent.py` → `ask` 命令 | 5-step pipeline: extract→build→identify→estimate→explain |

### Agent 新增命令

| 命令 | 说明 |
|------|------|
| `ask <自然语言问题>` | 一步完成因果分析：LLM 提取因果图 → 识别 → 估计 → 中文解读 |

### Bug修复

| 模块 | 文件 | 问题 | 修复 |
|------|------|------|------|
| do-calculus 死代码 | `core/identification.py` | R1-R3 已实现但 `identify_effect()` 从不调用；back-door/front-door/IV 失败后直接返回"不可识别" | 新增 Phase 4: do-calculus fallback — 穷举所有候选调整集 W，用 Rule 1 (Y ⊥ Z \| X,W in G_{X̄}) 验证，使之前被误判为不可识别的因果图现在可被识别 |

### 算法改进

| 改进 | 详细 |
|------|------|
| `identify_effect()` 管线升级 | 1.d-separation → 2.back-door → 3.front-door → **4.do-calculus** (新增) → 5.not identifiable |
| do-calculus Rule 1 搜索 | 穷举非后代变量的所有子集作为调整集，用 d-separation 在 G_{X̄} 中验证有效性 |

---

## v0.9.3 (2026-05-14) — 因果中介分析

### 新增模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 拉格朗日力学系统 | `core/physics.py` | LagrangianSystem, ActionPrinciple, 工厂函数 |
| 变分原理约束 | `core/physics.py` | ConstraintType.VARIATIONAL — 轨迹级物理约束 |
| 最小作用量 Demo | `demos/least_action_demo.py` | 单摆 + 谐振子轨迹验证 |

### 算法改进

| 改进 | 详细 |
|------|------|
| 物理定律库 14→15 条 | 新增 `least_action`: δS = δ∫L dt = 0 |
| 约束层级提升 | 单条定律 → 守恒律 → **变分原理** (全局轨迹约束) |
| ActionPrinciple | compute_action, validate_trajectory, find_stationary_path, compare_paths |
| Fibonacci search 梯度下降 | 支持从任意初始路径恢复物理稳态路径 |

### 新增系统

| 系统 | 拉格朗日量 | 运动方程 |
|------|-----------|---------|
| harmonic_oscillator | L = ½mẋ² − ½kx² | ẍ + (k/m)x = 0 |
| simple_pendulum | L = ½ml²θ̇² − mgl(1−cosθ) | θ̈ + (g/l)sinθ = 0 |

### 文档

| 文档 | 更新内容 |
|------|---------|
| PHYSICS_LEAST_ACTION.md | 完整设计文档 — 原理、代码、Demo、理论推导 |
| 全部 8 份文档 | 版本号升至 v0.9.2 |

---

## v0.9.1 (2026-05-14) — GES collider 修复 + 文档同步

### Bug修复

| 模块 | 文件 | 问题 | 修复 |
|------|------|------|------|
| GES collider 多余边 | `core/discovery.py` | collider X→Z←Y 下 BIC 贪心添加 X→Y 多余边（有限样本偶然相关） | Phase 3: CI 条件独立性剪枝，fisher_z_test 消除多余边 |

### 算法改进

| 改进 | 详细 |
|------|------|
| GES 从两阶段升级为三阶段 | Phase 1 (Forward) 贪心加边 → Phase 2 (Backward) 贪心删边 → **Phase 3 (CI Pruning)**: 对每条边尝试条件独立性验证，有分离集则删除 |

### 测试

| 类型 | 文件数 | 测试数 | 新增 |
|------|:---:|:---:|------|
| 单元测试 | 7 | 52 | test_ges_collider_pruning, test_ges_chain_preserved |

### 文档更新

| 文档 | 更新内容 |
|------|---------|
| ARCHITECTURE.md | 发现模块增加 GES Phase 3 描述 |
| MODELS_AND_ALGORITHMS.md | GES 算法描述更新为三阶段 |
| ROADMAP.md | 标记 GES collider 修复完成，版本升至 v0.9.1 |
| RUNNING.md | 添加 GES CI 剪枝行为说明 |

---

## v0.9-dev (2026-05-11) — Phase 4: LLM + 物理 + 非线性

### 新增模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 物理因果引擎 | `core/physics.py` | 14条定律, 4领域, PhysicsInformedSCM |
| 时间序列发现 | `core/ts_discovery.py` | Granger, TS-PC, PCMCI-lite |
| 非线性SCM | `core/scm.py` | nonlinear_scm(), nonlinear_eq(), multiplicative_noise_eq() |

### Bug修复

| 模块 | 文件 | 问题 | 修复 |
|------|------|------|------|
| PC 环路 | `core/discovery.py` | 无向边定向产生环路 | Kahn拓扑排序引导 + 索引回退 |
| FCI 规则 | `core/discovery.py` | 仅R1/R2/R4 | 补全R1-R7 (Zhang 2008) |
| 反事实传播 | `core/scm.py` | 干预只传播到直接子节点 | 全拓扑序重算，递归传播到孙节点 |

### 测试

| 类型 | 文件数 | 测试数 |
|------|:---:|:---:|
| 单元测试 | 7 | 54 |

### 新增模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 物理因果引擎 | `core/physics.py` (900行) | PhysicsLaw, PhysicsLibrary, PhysicsInformedCausalGraph, PhysicsInformedSCM, SymbolicPhysicsDiscovery, physics_causal_pipeline |
| 物理扩展指南 | `PHYSICS_EXTENSION_GUIDE.md` | 三步扩展流程, formula编写规则, 完整代码模板 |

### 修改模块

| 模块 | 文件 | 变更 |
|------|------|------|
| FCI 定向规则 | `core/discovery.py` | 补全 R1-R7 (原仅 R1/R2/R4)，增加详细注释和Zhang(2008)引用 |
| PC 环路修复 | `core/discovery.py` | 无向边定向改用Kahn拓扑排序引导，替代贪心法，增加最终环路检测+回退 |

### 新增 Demo

| Demo | 文件 | 功能 |
|------|------|------|
| 全功能验证套件 | `demos/run_all.py` | 5 个核心测试：Simpson全流程/因果发现/反事实/现代方法/领域迁移 |
| LLM 集成原型 | `demos/llm_prototype.py` | 模拟LLM接入：自由文本→因果图→解读→建议 |
| 物理因果演示 | `demos/physics_causal_demo.py` | 三层物理约束：DAG边/SCM方程/守恒律验证 |
| 初学者教程 | `demos/tutorial.py` | 7步交互教程：从因果图到反事实 |
| 飞机消失预测 | `demos/aircraft_disappearance.py` | 从 time_series/ 移入 (原核心模块降级为demo) |

### 文档更新

| 文档 | 变更 |
|------|------|
| `ARCHITECTURE.md` | 新增 demos/目录, physics.py, PHYSICS_EXTENSION_GUIDE.md; 移除 time_series/ |
| `MODELS_AND_ALGORITHMS.md` | 新增第10章(物理因果引擎) + 第11章(模型与源文件对照表), 37处文件引用 |
| `ROADMAP.md` | 版本 v0.8→v0.9-dev, 进度条 70%→85%, Phase 4标记 🔧 |
| `RUNNING.md` | 新增 demos/ 运行命令, 教程入口, 物理引擎入口 |
| `DESIGN_REPORT.md` | 新增设计白皮书: 哲学/架构/原理/决策/边界 |
| `PHYSICS_EXTENSION_GUIDE.md` | 新建: 14条定律,4领域, 三步扩展流程 |

### 迁移

| 从 | 到 | 原因 |
|----|----|------|
| `time_series/aircraft.py` | `demos/aircraft_disappearance.py` | 非核心架构组件，改为demo |

---

## v0.8 (2026-05-11) — Phase 3: 现代因果推断方法

### 新增模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 现代因果方法 | `core/modern.py` (600行) | DML(双重机器学习), CATE(S/T/X-learner), CausalForest, do-why集成 |
| 轻量ML模型 | `core/modern.py` | RidgeRegression, PolynomialRidge (零外部依赖) |

### 新增算法

| 算法 | 函数 | 原理 |
|------|------|------|
| Double ML | `estimate_ate_dml()` | K-fold交叉拟合, Neyman正交得分, √n一致性 |
| S-learner CATE | `estimate_cate_slearner()` | 单模型 Y~f(T,X), CATE=f(1,x)-f(0,x) |
| T-learner CATE | `estimate_cate_tlearner()` | 双模型 μ₁(X), μ₀(X) |
| X-learner CATE | `estimate_cate_xlearner()` | 交叉伪效应 + 倾向加权 |
| Causal Forest | `estimate_cate_forest()` | SimpleCausalTree + SimpleCausalForest (B树集成) |
| do-why 后端 | `estimate_ate_dowhy()` | 可选外部引擎, 回退到本地 |

---

## v0.7 (2026-05-11) — Phase 2: 高级因果发现

### 新增模块

| 模块 | 文件 | 说明 |
|------|------|------|
| PAG 图类型 | `core/discovery.py` | PAGEdge (6种边标记), PAG (部分祖先图) |
| FCI 算法 | `core/discovery.py` | 隐变量容忍, Possible-D-SEP重测试 |
| 自举置信度 | `core/discovery.py` | bootstrap_edge_confidence(), 边频率 → 置信度 |

### 修改模块

| 模块 | 文件 | 变更 |
|------|------|------|
| PC 算法 | `core/discovery.py` | 增强: v-structure定向, Meek R1-R3, 环路保护 |

### 新增数据集

| 类型 | 说明 |
|------|------|
| Type 3 (干预推理) | 150 个问题, metadata格式 |

---

## v0.6 (2026-05-11) — Phase 1: 补全推理管线

### 新增模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 效应估计器 | `core/estimation.py` (720行) | Linear, PSM, IPW, DR, Stratification (5种) |
| 敏感性分析 | `core/sensitivity.py` (230行) | Rosenbaum bounds, E-value |
| DAG 可视化 | `core/visualization.py` (200行) | ASCII, DOT, Mermaid, PNG(可选) |

### Bug 修复

| 模块 | 文件 | 问题 | 修复 |
|------|------|------|------|
| SCM 反事实 | `core/scm.py` | noise作为位置参数传入，`eq.func(*args, noise_val)` 被忽略 | 改为关键字参数 `eq.func(*args, noise=noise_val)`, 修复所有调用点(6处) |

---

## v0.5 (初始版本) — 核心引擎

### 初始模块

| 模块 | 文件 | 核心能力 |
|------|------|---------|
| 因果图 | `core/graph.py` | CausalDAG, d-separation (道德图法), 拓扑排序 |
| 效应识别 | `core/identification.py` | Back-door, Front-door, IV, do-calculus三规则 |
| 结构因果模型 | `core/scm.py` | SCM, do()干预, counterfactual三阶段 |
| 因果发现 | `core/discovery.py` | PC算法, GES算法, Fisher z-test, G²检验 |
| NL解析 | `nlp/parser.py` | CausalParser, 模板库(5个场景) |
| 交互入口 | `agent.py` | CLI REPL, 命令解析, demo模式 |

### 初始数据集

| 类型 | 规模 | 说明 |
|------|------|------|
| Type 1 (结构学习) | 100 图 × 500 样本 | 随机DAG生成 |
| Type 2 (效应估计) | 300 问题 × 1000 样本 | ATE已知真值 |
| Type 4 (反事实) | 96 个三元组 | JSONL格式 |
| Type 5 (领域迁移) | 8 领域 × 1000 样本 | 同构因果结构 |

---

## 版本总览

```
v0.5    核心引擎      ████████░░░░░░░░░░░░░░░░  20%
v0.6    Phase 1 完成  ████████████████░░░░░░░░  40%  估计+反事实+敏感性+可视化
v0.7    Phase 2 完成  ████████████████████░░░░  55%  FCI+自举置信度
v0.8    Phase 3 完成  ████████████████████████  70%  DML+CATE+do-why
v0.9-dev Phase 4 就绪 ████████████████████████  85%  物理引擎+LLM原型+FCI补全
```

---

## 当前代码统计

| 类别 | 文件数 | 总行数 | 说明 |
|------|:---:|:---:|------|
| 核心引擎 (core/) | 9 | ~5,200 | graph, identification, scm, discovery, estimation, modern, sensitivity, visualization, physics |
| NLP | 1 | ~300 | parser |
| Agent | 1 | ~560 | 交互式入口 |
| Demo | 5 | ~2,200 | run_all, tutorial, llm_prototype, physics_causal_demo, aircraft_disappearance |
| 数据工具 | 2 | ~700 | training_data, generate_all_datasets |
| 数据集 | 955 | — | Type 1-5 |
| 文档 | 8 | ~3,500 | 架构/算法/设计/路线图/运行/数据/物理/变更 |
| **总计** | **981+** | **~12,500** | — |

# Causal Agent — 完善路线图

> 当前版本: v0.9.7  |  状态: Phase 4 完成 + LLM 集成 + 自主诊断 | Phase 5 待开始
> 目标: 从「能识别因果效应」到「能在真实场景中自主推理」

---

## 总览：六阶段递进

```
                当前 ██████████████████░░░░░░░░░░  72%
                      │
Phase 1 ─ 补全推理  ████████████████████████████  100%  ✅ 完成
Phase 2 ─ 高级发现  ████████████████████████████  100%  ✅ 完成
Phase 3 ─ 现代方法  ████████████████████████████  100%  ✅ 完成
Phase 4 ─ LLM 融合  ████████████████████████████  100%  ✅ 完成
Phase 5 ─ 产品化    ██░░░░░░░░░░░░░░░░░░░░░░░░░░    5%  (未开始)
Phase 6 ─ 前沿探索  ████████░░░░░░░░░░░░░░░░░░░░   30%  (原型中)
```

---

## Phase 1: 补全推理管线 (P0) ✅ 已完成

> **完成日期**: 2026-05-11
> **新增文件**: `core/estimation.py`, `core/sensitivity.py`, `core/visualization.py`
> **修复**: `core/scm.py` 反事实 noise 关键字参数 bug

### 1.1 效应估计模块 `core/estimation.py`

| 估计器 | 适用场景 | 难度 | 输入 | 输出 |
|--------|---------|------|------|------|
| LinearRegression | 线性 SCM, 已知调整集 | ★ | (data, T, Y, Z) | ATE ± CI |
| PropensityScoreMatching | 二值处理, 强可忽略假设 | ★★ | (data, T, Y, X) | ATT, ATE |
| InverseProbabilityWeighting | 连续/二值处理 | ★★ | (data, T, Y, X) | ATE, 权重诊断 |
| DoublyRobust | 结合 PSM + 回归, 双保险 | ★★★ | (data, T, Y, X) | ATE, 对模型误设鲁棒 |
| Stratification | 离散混杂因子 | ★ | (data, T, Y, strata) | 分层 ATE |

```
实现要点:
  - 每个估计器返回 (estimate, std_error, confidence_interval)
  - 协变量平衡检查 (SMD < 0.1)
  - 倾向得分重叠检查 (common support)
  - 与 identification.py 无缝对接: 识别结果 → 调整集 → 估计
```

### 1.2 反事实推理修复

```
当前问题:
  - counterfactual() 对中间变量的更新不完整
  - 只支持线性 SCM

修复计划:
  1. 重写 counterfactual(): 正确传播干预对下游变量的影响
  2. 添加非参数反事实 (基于 SCM 采样的蒙特卡洛方法)
  3. 支持多个同时干预
  4. 添加个体处理效应 (ITE) 估计
```

### 1.3 敏感性分析

```
功能: "如果存在未观测的混杂因子 U, 它对结论有多大的威胁?"

实现:
  - Rosenbaum bounds (二值处理)
  - Cinelli-Hazlett 方法 (连续处理, 基于偏 R²)
  - E-value (VanderWeele-Ding)

输出示例:
  "要推翻'治疗有效'的结论, 未观测混杂因子需要
   将治疗分配概率改变至少 2.3 倍, 或将结果改变 1.5 个标准差"
```

### 1.4 DAG 可视化

```
实现:
  - 使用 graphviz/pygraphviz 生成 DAG 图片
  - 高亮显示: treatment (红), outcome (蓝), 调整集 (绿虚线框)
  - 标注后门路径
  - 导出 Mermaid (已有) + PNG/SVG
  
集成到 agent:
  > dag show     → 生成并显示当前 DAG 图片
  > dag save png → 保存到文件
```

---

## Phase 2: 高级因果发现 (P1) ✅ 已完成

> **完成日期**: 2026-05-11
> **新增**: `PAG` 类, `fci_algorithm()`, `bootstrap_edge_confidence()`

### 2.1 FCI 算法 (处理隐变量)

```
PC 的局限: 假设没有隐变量 (causal sufficiency)
FCI (Fast Causal Inference): 允许隐变量的存在

FCI 输出: PAG (Partial Ancestral Graph)
  边类型: →  (directed)
          ◦→ (ancestral, possibly indirect)
          ◦—◦ (undetermined)
          ↔  (confounded — 隐变量!)

实现:
  - PC 骨架 → 额外条件独立性检验 → FCI 定向规则
  - 输出 PAG (新的图类型, 需要扩展 CausalDAG)
```

### 2.2 自举法 + 边置信度

```
问题: PC 的输出对 alpha 阈值敏感, 单次运行可能有随机误差

方案: Bootstrap PC
  1. 对数据重采样 B 次 (B=100)
  2. 每次运行 PC
  3. 统计每条边出现的频率 → 边置信度

输出:
  边 X→Y: 置信度 0.87  ████████████████░░░
  边 X→Z: 置信度 0.42  ████████░░░░░░░░░░░░
```

### 2.3 其他发现算法

```
LiNGAM: 线性非高斯无环模型
  - 利用非高斯性 (ICA) 识别因果方向
  - 可以区分 Markov 等价类中的方向
  - 输出: 完全定向的 DAG (非等价类)

NOTEARS: 连续优化方法
  - 将 DAG 约束转化为可微的矩阵约束
  - 适合与神经网络结合
  - 可扩展到非线性 (用 MLP)

Granger Causality (时间序列):
  - X Granger-causes Y: X 的过去值有助于预测 Y 的未来值
  - 适用于时间序列的 aircraft 场景
```

---

## Phase 3: 现代因果推断方法 (P1) ✅ 已完成

> **完成日期**: 2026-05-11
> **新增文件**: `core/modern.py`

### 3.1 双重机器学习 (Double ML)

```
核心思想: 用 ML 模型估计 nuisance functions, 然后正交化

算法:
  1. E[Y|X] = g(X)     ← 任意 ML 模型
  2. E[T|X] = m(X)     ← 任意 ML 模型
  3. Y_res = Y - g(X), T_res = T - m(X)
  4. ATE = (Y_res · T_res) / (T_res · T_res)

优势:
  - 不需要知道 Y 和 T 的真实函数形式
  - 收敛速度达到 √n (即使 g, m 收敛更慢)
  - n=2000 即可获得可靠估计

实现:
  - 内置: 线性 + Lasso, 随机森林, XGBoost
  - 非线性 ATE: Causal Forest
```

### 3.2 异质性处理效应 (CATE)

```
问题: ATE 是平均效应, 但不同人群的效应可能不同

方法:
  Causal Forest (Athey & Imbens, 2016):
    - 基于随机森林
    - 估计每个人的个体处理效应
    - 输出: CATE(x) for each x

  Meta-learners:
    - S-learner: 单一模型 Y ~ f(T, X)
    - T-learner: 两个模型 μ₁(X), μ₀(X) → CATE = μ₁ - μ₀
    - X-learner: 对 T-learner 的改进

输出:
  "平均效应 ATE = 2.3, 但对年轻人 (age<30) 效应=5.1,
   对老年人 (age>60) 效应=-0.3 (可能有害)"
```

### 3.3 工具变量 + 深度学习

```
Deep IV (Hartford et al., 2017):
  - 当 IV 满足条件时用于估计非线性因果效应
  - 两阶段: 1) 深度网络估计 E[T|Z,X]
            2) 深度网络估计 E[Y|Ê[T|Z,X], X]

应用场景: 推荐系统, 定价优化, 政策评估
```

### 3.4 do-why 集成

```
do-why (Microsoft): 最成熟的因果推断 Python 库

集成策略: 作为后端引擎之一, 不替代核心

  identification → do-why 或 本地引擎
  estimation → do-why (丰富的估计器) 或 本地
  refutation → do-why (敏感性分析、安慰剂检验)

命令:
  > backend dowhy     → 切换到 do-why
  > backend native    → 切换到本地引擎
```

---

## Phase 4: LLM 融合 (P2) 🔧 原型就绪

> **原型**: `demos/llm_prototype.py` — 展示完整 LLM 集成流程
> **待完成**: 接入真实 LLM API (OpenAI / Anthropic / Ollama)
> **新增**: 物理规律集成原型 `demos/physics_causal_demo.py`

### 4.1 LLM 驱动的因果图提议

```
输入 (自然语言):
  "我想知道大学教育是否增加收入。我怀疑家庭背景
   同时影响教育机会和收入水平。另外，个人能力可能
   影响进入好大学的机会。请帮我构建因果图。"

LLM 输出:
  {
    "variables": ["FamilySES", "Ability", "Education", "Income"],
    "edges": [
      ["FamilySES", "Education"],
      ["FamilySES", "Income"],
      ["Ability", "Education"],
      ["Education", "Income"]
    ],
    "confounders": [["FamilySES"], ["Ability"]],
    "justification": "FamilySES is a confounder because..."
  }

人类验证: 展示图 → 确认/修改 → 锁定 DAG
```

### 4.2 自然语言结果解释

```
输入 (分析结果):
  ATE = 2.3, CI = [1.8, 2.9], 调整集 = {SES, Ability}

LLM 生成:
  "根据我们的分析，大学教育使收入平均增加 2.3 万元/年
   (95% 置信区间: 1.8-2.9 万元)。
   这个估计已经调整了家庭背景和个人能力的影响——
   也就是说，即使两个人在家庭背景和能力上完全相同，
   接受大学教育的那个人预期收入会高出 2.3 万元。"
```

### 4.3 反事实解释生成

```
输入:
  observed: {Education: 16, Income: 8, SES: "low"}
  intervention: {Education: 12}
  counterfactual_income: 5.2

LLM 生成:
  "如果一个低收入家庭出身的人，实际上读了 16 年书
   (大学毕业)，年收入是 8 万元。
   但在反事实世界里——假设他/她只读了 12 年书
   (高中毕业)——预期年收入将下降到 5.2 万元。
   也就是说，大学教育为这个人带来了约 2.8 万元的
   收入溢价。"
```

---

## Phase 5: 产品化 (P2)

### 5.1 Web 交互界面

```
技术栈: FastAPI + React/Vue

核心页面:
  1. DAG 编辑器 (拖拽画图, 或自然语言生成)
  2. 数据上传 (CSV, 数据库连接)
  3. 分析面板 (识别 → 估计 → 反事实, 一键完成)
  4. 报告生成 (自动生成 PDF 因果分析报告)
  5. 历史记录 (每次分析的完整快照)
```

### 5.2 API Server

```
REST API:
  POST /analyze
    body: {dag, data, treatment, outcome}
    response: {ate, ci, adjustment_set, warning_flags}

  POST /discover
    body: {data, method: "pc"|"ges"|"fci"}
    response: {dag, edge_confidences, warnings}

  POST /counterfactual
    body: {scm, observed, intervention, target}
    response: {counterfactual_value, explanation}
```

### 5.3 部署 + 监控

```
Docker 化: docker-compose up → 全栈启动

监控指标:
  - 因果效应估计的稳定性 (滚动窗口 CUSUM)
  - 倾向得分分布漂移 (PS distribution shift)
  - 数据分布漂移 (covariate shift → 模型需重新标定)

因果监控仪表板:
  "上周的效应估计: 2.3 → 这周: 1.8, 下降了 22%,
   值得调查原因 (数据漂移? 真实变化?)"
```

---

## Phase 6: 前沿探索 (P3)

### 6.1 时间序列因果发现

```
当前: aircraft.py 的因果图是手动指定的

目标: 从时间序列数据中自动发现因果结构

算法:
  - PCMCI (Runge et al., 2019): PC + 瞬时条件独立性
  - VAR-LiNGAM: 向量自回归 + 非高斯性
  - TCDF (Nauta et al., 2019): 时间卷积 + 注意力

应用: 金融风控, 工业预测性维护, 气候归因
```

### 6.2 因果表示学习

```
目标: 从高维数据 (图像, 文本) 中学习因果表示

方法:
  - β-VAE: 解耦表示学习
  - CausalVAE: 用因果图约束 VAE 的隐空间
  - CITRIS: 因果识别的过渡状态表示

应用: 医学影像的因果特征提取, 公平机器学习
```

### 6.3 强化学习 + 因果推断

```
场景: 智能体不仅推断因果, 还主动设计实验来发现因果

方法:
  - Causal Bandit: 每一步选择一个干预, 学习最优干预
  - Active Causal Discovery: 主动选择实验来减少 DAG 不确定性
  - Counterfactual RL: 用反事实推理改进策略学习

应用到 agent:
  > "对于当前的因果图, 哪几个干预实验能最快
     消除剩余的因果方向不确定性?"
```

### 6.4 因果公平性

```
问题: ML 模型可能通过代理变量歧视受保护群体

方法:
  - Counterfactual Fairness: 反事实世界里敏感属性不应改变预测
  - Path-specific Fairness: 只允许通过"公平"路径的影响
  - Causal Fairness Analysis: 检测哪些决策路径引入了不公平

输出:
  "模型预测中, 性别通过'职业选择'路径的间接影响
   占总效应的 12%, 这可能是公平性问题。建议:
   1) 在'职业'变量上做路径特定的公平约束
   2) 或删除'职业'变量, 仅保留直接影响"
```

---

## 实施优先级矩阵

```
                    影响力
                低        中        高
           ┌─────────┬─────────┬─────────┐
费    低   │1.4 DAG  │1.3 敏感 │1.1 估计 │
          │可视化   │性分析   │模块     │
          ├─────────┼─────────┼─────────┤
用    中   │3.3 Deep│2.1 FCI  │1.2 反事 │
          │   IV    │         │实修复   │
          ├─────────┼─────────┼─────────┤
     高   │2.3 其他 │2.2 自举 │3.1 DML  │
          │发现算法 │         │3.2 CATE │
          └─────────┴─────────┴─────────┘

推荐顺序: 1.1 → 1.2 → 1.3 → 3.1 → 2.1 → 3.2 → 5.1 → 4.1
         (补全推理) (反事实) (敏感性) (DML) (FCI) (CATE) (UI) (LLM)
```

---

## 里程碑

| 版本 | 内容 | 状态 |
|------|------|------|
| v0.6 | Phase 1 完成 (估计 + 反事实修复 + 敏感性 + 可视化) | ✅ |
| v0.7 | Phase 2 (FCI + 自举) | ✅ |
| v0.8 | Phase 3 (DML + CATE + do-why) | ✅ |
| v0.9.1 | Phase 4 (LLM 原型 + 物理集成 + GES collider 修复) | ✅ |
| v0.9.2 | Phase 4+ (最小作用量原理集成) | ✅ |
| v0.9.3 | Phase 4+ (因果中介分析: NDE/NIE/CDE) | ✅ |
| v0.9.4 | Phase 4+ (P1修复: do-calculus 接入管线) | ✅ |
| v0.9.5 | Phase 4+ (DeepSeek LLM 集成 — 自然语言接口) | ✅ |
| v0.9.6 | Phase 4+ (自主诊断 + 自动数据生成 + 自动方法选择) | ✅ |
| v1.0 | Phase 5 (Web UI + API + Docker) | |
| v1.1+ | Phase 6 前沿探索 | |

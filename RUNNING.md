# Causal Agent — 运行指南

> 最后更新: 2026-05-18
> 版本: v0.9.10 | 日期: 2026-05-18
> Python 版本: ≥ 3.10

---

## 一、环境准备

### 1.1 系统依赖

```bash
# Arch Linux
sudo pacman -S python python-numpy

# Ubuntu / Debian
sudo apt install python3 python3-pip python3-venv
```

### 1.2 虚拟环境

```bash
python3 -m venv ~/causal_venv
source ~/causal_venv/bin/activate
pip install numpy
```

### 1.3 命令历史（↑↓ 导航）

交互模式下支持 ↑↓ 键浏览前序输入的命令：

```
↑ 键 — 上一条命令
↓ 键 — 下一条命令
```

- 自动保存到 `~/.hermes/causal_agent_history`（最多 2000 条）
- 跨 session 持久化 — 关闭后下次启动仍可回溯
- 自动去重 — 连续相同命令只保留一条
- 依赖 Python `readline` 模块（Linux 自带，Windows 需 `pip install pyreadline3`）

---

## 二、两种运行模式

| 模式 | 入口 | 适用场景 | 数据要求 |
|------|------|---------|---------|
| **LLM 模式** | `ask <自然语言问题>` | 快速探索、未知场景、非技术用户 | 无需准备 — Agent 自动生成合成数据 |
| **无 LLM 模式** | `load` / `discover` / `effect` 命令 | 精确控制、已有数据、算法研究 | 需 CSV 文件或模板名称 |

---

## 三、LLM 模式 — 自然语言直接提问

### 3.1 配置 API Key

```bash
# 方式一：环境变量（临时）
export DEEPSEEK_API_KEY="sk-你的key"

# 方式二：配置文件（永久，推荐）
mkdir -p ~/.hermes
echo '{"DEEPSEEK_API_KEY":"sk-你的key"}' > ~/.hermes/causal_config.json
chmod 600 ~/.hermes/causal_config.json
```

配置文件路径为 `~/.hermes/causal_config.json`，Agent 会自动读取，无需每次设置环境变量。

### 3.2 使用方式

```bash
cd /home/duyw/causal_agent
python agent.py
```

进入交互模式后：

```
> ask 吸烟会导致肺癌吗

  Agent 自动执行 5 步：
  Step 1: LLM 从自然语言提取因果图（变量、边、处理、结局、混杂因子）
  Step 2: 构建 CausalDAG
  Step 3: 识别因果效应（back-door / front-door / do-calculus）
  Step 4: 生成合成数据 → 诊断假设 → 自动选择估计方法 → 估计 ATE
  Step 5: LLM 生成中文自然语言解读
```

### 3.3 支持的提问类型

```
"吸烟会导致肺癌吗"
"教育年限对收入的影响有多大，控制家庭背景"
"锻炼是否降低心脏病风险"
"广告投放对销售额的因果效应，考虑季节因素"
"最低工资政策对就业率的影响"
```

### 3.4 LLM 模式下 Agent 的自主能力

| 能力 | 说明 |
|------|------|
| 自动变量提取 | 从自然语言中识别因果变量和方向 |
| 自动混杂因子推断 | 基于领域知识推断可能的混淆变量 |
| 自动数据生成 | 无数据时从因果图构建线性 SCM 并采样 500 条 |
| 自动假设诊断 | 检查残差正态性、协变量重叠性 |
| 自动方法选择 | linear → PSM → IPW → DR，根据诊断自动切换 |
| 自然语言解释 | 中文输出 ATE、CI、E-value、稳健性、行动建议 |

### 3.5 示例输出

```
> ask 吸烟会导致肺癌吗

Query: 吸烟会导致肺癌吗
──────────────────────────────────────────────────
Step 1: Extracting causal structure...
  Variables: Smoking, LungCancer, Genetics, AirPollution, Age
  Edges: Smoking→LungCancer, Genetics→Smoking, Genetics→LungCancer, ...
  Treatment: Smoking, Outcome: LungCancer
  Confounders: Genetics, Age

Step 2: Building causal model...
  DAG built: Causal DAG: 5 variables, 6 edges

Step 3: Identifying causal effect...
  Method: Back-door adjustment
  Adjustment set: {Age, Genetics}
  Identifiable: True

Step 4: Estimating causal effect...
  No data loaded — generating synthetic data from SCM...
  Generated 500 synthetic samples from linear SCM

  Diagnostics:
  linearity ✓, overlap ✓
  → Selected: linear

  ATE = 1.5379  (SE = 0.0933)
  95% CI = [1.3552, 1.7207]
  ✓ significant

  E-value: 8.78  (Highly robust)

Step 5: Natural language explanation...
  吸烟确实会显著增加患肺癌的风险...
  [详细中文解读：效应大小、置信区间含义、混杂控制、稳健性评估、行动建议]
```

---

## 四、无 LLM 模式 — 手动操作

适用于：已有结构化数据、需要精确控制分析流程、离线环境。

### 4.1 方式一：使用预置模板

Agent 内置 5 个预置因果场景模板，无需 LLM，无需数据。

```bash
> load simpson        # Simpson's Paradox (新药→康复，混淆：性别)
> load smoking        # 吸烟与肺癌（含遗传混杂）
> load education      # 教育对收入的影响
> load frontdoor      # 前门调整示例
> load mbias          # M-bias 对撞结构

# 加载后立即分析
> effect D R          # 估计 D 对 R 的因果效应
> whatif D=0 R        # 干预推演：如果 D=0，R 会是多少？
> dag show            # ASCII 图展示 DAG
> sensitivity D R     # 敏感性分析
```

### 4.2 方式二：自由文本构建因果模型

无需 LLM，用规则解析器从结构化描述中提取因果图：

```bash
> load variables: X, Y, Z; edges: X→Z, Y→Z
> load X causes M; M causes Y; X also causes Y
> load treatment is Education; outcome is Income; confounders: FamilySES, Ability
```

### 4.3 方式三：从数据中发现因果结构

**Step 1: 准备 CSV 数据文件**

```csv
# example: simpson_data.csv
Gender,Drug,Recovery
F,1,1
M,1,1
F,0,1
M,0,0
F,1,1
...
```

要求：
- 第一行为表头（变量名）
- 逗号分隔
- 数值型数据
- 建议 500+ 行样本

**Step 2: 加载数据并发现因果结构**

```bash
> load_data /path/to/data.csv
  Loaded 1000 samples, 4 variables: Gender, Drug, Recovery, Age

# PC 算法（快，假设无隐变量）
> discover /path/to/data.csv pc

# FCI 算法（允许隐变量，检测混淆边）
> discover /path/to/data.csv fci

# GES 算法（基于 BIC 评分，含 CI 剪枝）
> discover /path/to/data.csv ges

# 带自举置信度
> discover /path/to/data.csv pc --bootstrap=100

# 指定 α 水平（默认 0.05）
> discover /path/to/data.csv pc --alpha=0.01
```

**Step 3: 效应估计**

```bash
# 自动选择方法
> effect Drug Recovery

# 指定方法
> effect Drug Recovery linear    # 线性回归
> effect Drug Recovery dr        # 双重鲁棒
> effect Drug Recovery ipw       # 逆概率加权
> effect Drug Recovery psm       # 倾向得分匹配
> effect Drug Recovery dml       # 双重机器学习
```

**Step 4: 敏感性分析**

```bash
> sensitivity Drug Recovery
```

### 4.4 数据准备速查表

| 数据类型 | 格式 | 最低样本 | 示例 |
|---------|------|:---:|------|
| 因果发现 | CSV, 首行表头, 数值列 | 500 | `discover data.csv pc` |
| 效应估计 | CSV + 已加载 DAG | 300 | `effect X Y linear` |
| 反事实推理 | SCM 参数 + 观测值 | N/A | `whatif X=0 Y given X=1,Y=2` |
| 时间序列因果 | CSV, 多变量, 等间距 | 200 | `granger_test(data, vars)` |

### 4.5 完整无 LLM 工作流示例

```bash
cd /home/duyw/causal_agent
python agent.py

# 1. 加载预置场景
> load simpson
  Loaded template: simpsons_paradox

# 2. 查看因果图
> dag show
  ┌───┐
  │ G │────→ D ────→ R
  └───┘         ↗
    └───────────┘
  G=Gender, D=Drug, R=Recovery

# 3. 识别因果效应
> effect D R
  Method: Back-door adjustment
  Adjustment set: {G}
  
  Causal Effect Estimate:
  ATE = 0.3124  (SE = 0.0456)
  95% CI = [0.2231, 0.4017]
  ✓ significant

# 4. 敏感性分析
> sensitivity D R
  E-value: 5.2  (Highly robust)

# 5. 反事实
> whatif D=0 R given G=1,D=1,R=1
  Counterfactual: if D had been 0 instead of 1, R would be 0.68
```

---

## 五、从 CSV 数据构建因果模型的完整流程

如果你有一份 CSV 数据但不知道因果结构，走这条路径：

```
CSV 数据
  │
  ├─→ discover data.csv pc     (自动发现因果结构)
  │      ↓
  │   CausalDAG
  │      ↓
  ├─→ effect X Y               (识别 + 估计)
  │      ↓
  │   ATE ± CI + 敏感性
  │
  └─→ whatif X=0 Y             (干预推演)
```

---

## 六、运行演示

```bash
cd /home/duyw/causal_agent

# 全功能验证（5 个 demo）
python demos/run_all.py

# 初学者教程（7 步循序渐进）
python demos/tutorial.py

# 物理因果推断
python demos/physics_causal_demo.py

# 最小作用量原理
python demos/least_action_demo.py

# 运行全部测试（52 个）
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

## 七、命令速查

| 命令 | 说明 | 示例 |
|------|------|------|
| `ask <问题>` | 自然语言提问（需 LLM） | `ask 吸烟会导致肺癌吗` |
| `load <描述>` | 加载场景/模板 | `load simpson` |
| `discover <csv> [pc\|fci\|ges]` | 从数据发现因果结构 | `discover data.csv pc` |
| `effect <X> <Y> [方法]` | 估计因果效应 | `effect D R dr` |
| `whatif <X=v> <Y>` | 干预推演 | `whatif D=0 R` |
| `whatif ... given ...` | 反事实推理 | `whatif D=0 R given D=1,R=1` |
| `sensitivity <X> <Y>` | 敏感性分析 | `sensitivity D R` |
| `dag show` | ASCII 图展示 DAG | |
| `dag save png /tmp/dag.png` | 导出 DAG 图 | |
| `model` | 查看当前模型 | |
| `explain <概念>` | 解释因果概念 | `explain backdoor` |
| `demo` | 运行演示 | |
| `help` | 显示帮助 | |
| `quit` | 退出 | |

---

## 八、常见问题

### Q: "LLM extraction failed: DEEPSEEK_API_KEY not set"
A: API key 未配置。创建 `~/.hermes/causal_config.json`：
```bash
echo '{"DEEPSEEK_API_KEY":"sk-你的key"}' > ~/.hermes/causal_config.json
```

### Q: "No scenario loaded"
A: 先 `load simpson` 或 `ask 你的问题` 加载因果模型。

### Q: 数据格式要求
A: CSV，首行表头，数值列，建议 ≥300 行。PC 算法 ≥200 行，GES ≥500 行。

### Q: 如何获取 DeepSeek API Key
A: 访问 https://platform.deepseek.com 注册并获取。

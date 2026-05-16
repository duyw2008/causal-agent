# Causal Agent — Linux 运行指南

> 最后更新: 2026-05-14
> 版本: v0.9.2
> Python 版本: ≥ 3.10

---

## 一、环境准备

### 1.1 系统依赖

```bash
# Arch Linux
sudo pacman -S python python-numpy python-scipy nodejs npm

# Ubuntu / Debian
sudo apt install python3 python3-pip python3-venv nodejs npm

# CentOS / RHEL
sudo dnf install python3 python3-pip nodejs npm
```

### 1.2 创建虚拟环境

```bash
# 创建 venv
python3 -m venv ~/causal_venv

# 激活
source ~/causal_venv/bin/activate

# 安装 Python 依赖
pip install numpy scipy

# 验证
python -c "import numpy; import scipy; print('OK')"
# → OK
```

### 1.3 安装 Node.js 依赖 (PPT 生成用，可选)

```bash
npm install -g pptxgenjs
```

### 1.4 获取代码

```bash
# 项目目录
cd /home/duyw/causal_agent
```

**项目已在 `/home/duyw/causal_agent/`，无需额外克隆。**

---

## 二、快速启动

### 2.1 交互模式

```bash
cd /home/duyw/causal_agent
~/causal_venv/bin/python agent.py
```

```
============================================================
  Causal Inference Agent
  Type 'help' for commands, 'quit' to exit
============================================================

> help

Commands:
  load <description>   — describe a causal scenario
  template <name>      — load a pre-built template
  discover <file.csv>  — learn causal structure from data
  effect <X> <Y>       — identify effect of X on Y
  whatif <X=val> <Y>   — predict Y under intervention
  whatif <X=val> <Y> given <obs> — counterfactual
  explain [concept]    — explain a causal concept
  model                — show current DAG
  demo                 — run demonstrations
  quit                 — exit

Templates: smoking, simpson, education, frontdoor, mbias
```

### 2.2 运行演示

```bash
cd /home/duyw/causal_agent
~/causal_venv/bin/python demos/run_all.py
```

一次性运行 5 个核心 demo，验证全功能：
1. Simpson's Paradox — 识别→估计→敏感性分析
2. Causal Discovery — PC/FCI/Bootstrap 因果发现
3. Counterfactual — 反事实推理（吸烟→癌症）
4. Modern Methods — DML + CATE 9 种估计器对比
5. Domain Transfer — 8 领域 ATE 恢复

```bash
# LLM 集成原型
~/causal_venv/bin/python demos/llm_prototype.py

# 初学者交互教程
~/causal_venv/bin/python demos/tutorial.py

# 物理规律 + 因果推断原型
~/causal_venv/bin/python demos/physics_causal_demo.py
```

### 2.3 单次查询

```bash
cd /home/duyw/causal_agent
~/causal_venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from agent import CausalAgent
agent = CausalAgent()
print(agent.load_scenario('simpson'))
print(agent.ask_effect('D', 'R'))
"
```

---

## 三、交互命令详解

### 3.1 加载场景

```
# 自然语言描述
> load variables: Education, Income, SES. SES causes Education.
         SES causes Income. Education causes Income.

# 加载预置模板
> template simpson    # Simpson's Paradox
> template smoking    # Smoking → Tar → Lung Cancer
> template frontdoor  # Front-door adjustment
> template mbias      # M-bias (collider)
> template education  # 教育-收入
```

### 3.2 因果效应识别 + 估计

```
> effect Treatment Outcome

# Phase 1 新功能: 加载数据后进行数值估计
> load_data /path/to/data.csv
> effect D R
  Query: P(R | do(D))
  Method: Back-door adjustment
  Adjustment set: {G}

Causal Effect Estimate:
  ATE = -1.1112  (SE = 0.0227)
  95% CI = [-1.1556, -1.0667]
  Statistically significant ✓

SENSITIVITY ANALYSIS:
  Rosenbaum Bounds:
    Results are robust to hidden bias up to Γ = 5.0.
  E-value:
    Highly robust (E-value = 5.5).

# 选择估计方法
> effect D R linear    # 线性回归
> effect D R psm       # 倾向得分匹配
> effect D R ipw       # 逆概率加权
> effect D R dr        # 双重鲁棒
> effect D R stratified # 分层
> effect D R auto      # 自动选择 (默认)
```

### 3.3 DAG 可视化

```
> dag show          # ASCII 艺术画
> dag save png      # 导出 PNG (需 graphviz)
> dag save svg      # 导出 SVG
> dag save dot      # 导出 DOT 源文件
```

示例:
> effect D R
  Query: P(R | do(D))
  D is an ancestor of R — causal path exists
  Back-door adjustment set: {G}
  → Adjust for G to de-confound

> effect Education Income
  (自动计算调整集并给出可估计表达式)
```

### 3.3 干预预测 (what-if)

```
# 群体层面 (不需要观测数据)
> whatif X=1.0 Y
  Intervention: do(X=1.0)
  → E[Y] = 2.3456

# 反事实层面 (需要观测数据)
> whatif Smoking=0 Cancer given Smoking=3,Tar=2.5,Cancer=4,Gene=2
  Counterfactual:
  Observed: {Smoking:3, Tar:2.5, Cancer:4, Gene:2}
  Intervention: do(Smoking=0)
  → Cancer = 2.2   (如果这个人不吸烟, 预期患癌风险降低)
```

### 3.4 因果发现 (从数据学习)

```bash
# 先准备数据
cd /home/duyw/causal_agent
~/causal_venv/bin/python -c "
from core.discovery import generate_linear_data
from core.graph import CausalDAG
import numpy as np

dag = CausalDAG(['G','D','R'], [('G','D'),('G','R'),('D','R')])
data = generate_linear_data(dag, n_samples=3000, seed=42)
np.savetxt('/tmp/simpson_data.csv', data, delimiter=',',
           header='G,D,R', comments='')
print('Data saved to /tmp/simpson_data.csv')
"
```

```bash
# 在 agent 中使用
> discover /tmp/simpson_data.csv pc
  Loaded 3000 samples, 3 variables
  Running PC algorithm (alpha=0.05)...
  Removed G—R | {D}
  Discovered: CausalDAG(G→D, D→R)

> discover /tmp/simpson_data.csv ges
  (使用 GES 算法)
```

### 3.5 概念解释

```
> explain backdoor
> explain dseparation
> explain do
> explain scm
> explain frontdoor
```

---

## 四、运行独立模块

### 4.1 DAG + d-separation 测试

```bash
cd /home/duyw/causal_agent
~/causal_venv/bin/python core/graph.py
```

```
All DAG tests passed ✓
Causal DAG: 3 variables, 2 edges
  X:  (exogenous)  children: [M]
  M:  parents: [X]  children: [Y]
  Y:  parents: [M]
```

### 4.2 SCM + 干预 + 反事实测试

```bash
cd /home/duyw/causal_agent
~/causal_venv/bin/python core/scm.py
```

### 4.3 因果发现测试

```bash
cd /home/duyw/causal_agent/core
~/causal_venv/bin/python discovery.py
```

输出包含 PC 和 GES 在 Chain / Fork / Collider / Simpson 四个结构上的结果。

> **GES Phase 3**: 自 v0.9.1 起，GES 在 Phase 2 之后增加条件独立性(CI)剪枝阶段，自动消除 collider 结构 (X→Z←Y) 中因有限样本偶然相关产生的多余边 X→Y。

### 4.4 识别 (back-door/front-door/IV) 测试

```bash
cd /home/duyw/causal_agent
~/causal_venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from core.graph import CausalDAG
from core.identification import identify_effect

dag = CausalDAG(['G','D','R'], [('G','D'),('G','R'),('D','R')])
result = identify_effect(dag, 'D', 'R')
print(result)
"
```

### 4.5 自然语言解析测试

```bash
cd /home/duyw/causal_agent
~/causal_venv/bin/python nlp/parser.py
```

### 4.6 效应估计测试

```bash
cd /home/duyw/causal_agent
~/causal_venv/bin/python core/estimation.py
```

### 4.7 敏感性分析测试

```bash
cd /home/duyw/causal_agent
~/causal_venv/bin/python core/sensitivity.py
```

### 4.8 DAG 可视化测试

```bash
cd /home/duyw/causal_agent
~/causal_venv/bin/python core/visualization.py
```

### 4.6 飞机消失预测 (Demo)

```bash
cd /home/duyw/causal_agent
~/causal_venv/bin/python demos/aircraft_disappearance.py
```

输出:
- 5000 架次轨迹生成
- softmax 模型训练 (97.9% 准确率)
- 因果分析报告 (Weather-stratified probabilities)

### 4.7 生成训练数据集

```bash
cd /home/duyw/causal_agent
~/causal_venv/bin/python generate_all_datasets.py
```

生成 955 个文件到 `datasets/`，包含全部五类训练数据：

| 类型 | 路径 | 数量 |
|------|------|------|
| 结构学习 | `datasets/type1_structure_learning/` | 100 图 × CSV + JSON |
| 效应估计 | `datasets/type2_effect_estimation/` | 300 问题 × CSV + JSON |
| 干预推理 | `datasets/type3_interventional/` | 150 问题 |
| 反事实 | `datasets/type4_counterfactual/` | 96 三元组 (JSONL) |
| 领域迁移 | `datasets/type5_domain_transfer/` | 8 领域 × CSV + JSON |

---

## 五、Python API 使用

### 5.1 导入核心模块

```python
import sys
sys.path.insert(0, '/home/duyw/causal_agent')

from core.graph import CausalDAG
from core.identification import identify_effect
from core.scm import linear_scm
from core.discovery import pc_algorithm, generate_linear_data
```

### 5.2 端到端示例

```python
import sys
sys.path.insert(0, '/home/duyw/causal_agent')
import numpy as np
from core.graph import CausalDAG
from core.identification import identify_effect
from core.discovery import generate_linear_data, pc_algorithm

# 1. 生成数据 (已知真实因果结构)
true_dag = CausalDAG(['G','D','R'], [('G','D'),('G','R'),('D','R')])
data = generate_linear_data(true_dag, n_samples=2000, seed=42)

# 2. 因果发现 (从数据恢复结构)
learned_dag = pc_algorithm(data, ['G','D','R'], alpha=0.01)
print(f"True: {true_dag}")
print(f"Learned: {learned_dag}")

# 3. 效应识别
result = identify_effect(learned_dag, 'D', 'R')
print(f"\nEffect of D on R:")
print(f"  Identifiable: {result.identifiable}")
print(f"  Method: {result.method}")
print(f"  Adjustment set: {result.adjustment_set}")
```

### 5.3 使用训练数据

```python
import json
import numpy as np

# 加载 Type 2 问题
with open("datasets/type2_effect_estimation/index.json") as f:
    index = json.load(f)

problem = index["problems"][0]
pid = problem["problem_id"]

# 加载数据
data = np.genfromtxt(
    f"datasets/type2_effect_estimation/problem_{pid:04d}.csv",
    delimiter=',', skip_header=1
)

# 加载元数据
with open(f"datasets/type2_effect_estimation/problem_{pid:04d}_meta.json") as f:
    meta = json.load(f)

print(f"Treatment: {meta['treatment']}")
print(f"Outcome: {meta['outcome']}")
print(f"True ATE: {meta['true_ate']:.4f}")
print(f"Obs diff: {meta['obs_diff']:.4f}")
print(f"Confounding bias: {meta['confounding_bias']:.4f}")
```

---

## 六、常见问题

### Q: ModuleNotFoundError: No module named 'numpy'

```bash
~/causal_venv/bin/pip install numpy scipy
```

### Q: ImportError: attempted relative import

```bash
# 确保从项目根目录运行
cd /home/duyw/causal_agent

# 或在 Python 代码中添加
import sys
sys.path.insert(0, '/home/duyw/causal_agent')
```

### Q: PC 算法运行很慢

```bash
# 减小条件集大小上限 (默认无限制)
# 在代码中设置 max_cond_size=3
dag = pc_algorithm(data, var_names, alpha=0.05, max_cond_size=3)
```

### Q: 虚拟环境路径不同

```bash
# 使用你创建的 venv 路径, 不限于 ~/causal_venv
/path/to/your/venv/bin/python agent.py
```

---

## 七、文件索引

| 文件 | 用途 | 运行方式 |
|------|------|---------|
| `agent.py` | 交互式智能体 | `python agent.py` |
| `core/graph.py` | DAG + d-separation | `python core/graph.py` |
| `core/scm.py` | 结构因果模型 | `python core/scm.py` |
| `core/identification.py` | 因果效应识别 | 作为模块导入 |
| `core/discovery.py` | PC / GES 算法 | `python core/discovery.py` |
| `core/estimation.py` | 五类 ATE 估计器 | `python core/estimation.py` |
| `core/sensitivity.py` | 敏感性分析 | `python core/sensitivity.py` |
| `core/visualization.py` | DAG 可视化 | `python core/visualization.py` |
| `nlp/parser.py` | NL 解析器 | `python nlp/parser.py` |
| `demos/aircraft_disappearance.py` | 飞机消失预测 | `python demos/aircraft_disappearance.py` |
| `demos/tutorial.py` | 初学者交互教程 | `python demos/tutorial.py` |
| `training_data.py` | 训练数据生成器 API | 作为模块导入 |
| `demos/run_all.py` | 全功能验证套件 | `python demos/run_all.py` |
| `demos/llm_prototype.py` | LLM 集成原型 | `python demos/llm_prototype.py` |
| `demos/physics_causal_demo.py` | 物理规律集成 | `python demos/physics_causal_demo.py` |
| `core/physics.py` | 物理因果引擎 | `python core/physics.py` |
| `PHYSICS_EXTENSION_GUIDE.md` | 物理规律扩展指南 | 阅读 |
| `ARCHITECTURE.md` | 系统架构文档 | 阅读 |
| `DATA_REQUIREMENTS.md` | 数据需求方案 | 阅读 |
| `ROADMAP.md` | 六阶段路线图 | 阅读 |
| `datasets/README.md` | 数据集文档 | 阅读 |

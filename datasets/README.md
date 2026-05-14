# Causal Reasoning Training Datasets

> 自动生成时间: 2026-05-11
> 总文件数: 955  |  总大小: ~15.5 MB

---

## 目录结构

```
datasets/
├── README.md                          ← 本文档
├── type1_structure_learning/          ← Type 1: 结构学习
│   ├── index.json                     # 全局索引
│   ├── graph_0000.csv                 # 观测数据 (500 样本)
│   ├── graph_0000_meta.json           # 真实 DAG + v-structure
│   └── ... (100 个图)
├── type2_effect_estimation/           ← Type 2: 效应估计
│   ├── index.json
│   ├── problem_0000.csv               # 观测数据 (1000 样本)
│   ├── problem_0000_meta.json         # 真实 ATE + 观测差异 + 后门集
│   └── ... (300 个问题)
├── type3_interventional/              ← Type 3: 干预推理
│   ├── index.json
│   ├── problem_0000_meta.json         # E[Y|T] vs E[Y|do(T)]
│   └── ... (150 个问题)
├── type4_counterfactual/              ← Type 4: 反事实推理
│   ├── index.json
│   └── counterfactuals.jsonl          # 96 个反事实三元组
└── type5_domain_transfer/             ← Type 5: 领域迁移
    ├── index.json
    ├── medical.csv / medical_meta.json        # 医疗
    ├── economics.csv / economics_meta.json    # 经济
    ├── marketing.csv / marketing_meta.json    # 营销
    ├── education.csv / education_meta.json    # 教育
    ├── sports.csv / sports_meta.json          # 体育
    ├── software.csv / software_meta.json      # 软件
    ├── climate.csv / climate_meta.json        # 气候
    └── psychology.csv / psychology_meta.json  # 心理
```

---

## Type 1: 结构学习 (Structure Learning)

**训练目标**: 从纯观测数据推断因果图结构。

**数据格式**:

`graph_NNNN.csv` — 观测数据矩阵:
```
X0,X1,X2,X3
0.123,-0.456,0.789,-0.321
...
```
- 500 样本, 3-8 个变量
- 每个图由一个随机线性 SCM 生成

`graph_NNNN_meta.json`:
```json
{
  "graph_id": 0,
  "n_vars": 4,
  "n_samples": 500,
  "var_names": ["X0", "X1", "X2", "X3"],
  "edges": [["X0", "X2"], ["X1", "X3"]],
  "adj_matrix": [[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]],
  "v_structures": [],
  "noise_std": 0.35
}
```

**训练任务**:
- 输入: `graph_NNNN.csv`
- 输出: 因果边列表 (与 `edges` 比较)
- 指标: SHD (Structural Hamming Distance), F1-score on edges

---

## Type 2: 效应估计 (Effect Estimation)

**训练目标**: 给定因果图和观测数据，正确估计平均因果效应 (ATE)。

**数据格式**:

`problem_NNNN.csv` — 观测数据矩阵 (同 Type 1 格式)

`problem_NNNN_meta.json`:
```json
{
  "problem_id": 5,
  "var_names": ["X0", "X1", "X2", "X3"],
  "edges": [["X0","X1"],["X1","X2"],["X0","X2"]],
  "treatment": "X0",
  "outcome": "X2",
  "true_ate": 1.234,           ← 真实因果效应
  "obs_diff": 1.567,           ← 朴素观测差异 (有偏!)
  "confounding_bias": 0.333,   ← 偏差 = obs_diff - true_ate
  "has_confounding": true,
  "backdoor_set": ["X1"],      ← 需要调整的变量
  "n_vars": 4
}
```

**训练任务**:
- 输入: `problem_NNNN.csv` + DAG (从 edges 重建)
- 输出: ATE 估计值
- 指标: MAE(estimated_ATE, true_ate)

**关键洞察**: 31/300 的问题有混杂因子。
  观测差异 ≠ 因果效应。偏差范围 [0.04, 2.46]。

---

## Type 3: 干预推理 (Interventional Reasoning)

**训练目标**: 预测在 do() 干预下的结果分布。

**数据格式**:

`problem_NNNN_meta.json`:
```json
{
  "problem_id": 0,
  "var_names": ["X0","X1","X2"],
  "edges": [["X0","X2"],["X1","X2"]],
  "treatment": "X0",
  "outcome": "X2",
  "true_ate": 1.5,
  "obs_EY_given_T_low": 0.82,    ← 观测条件期望
  "obs_EY_given_T_high": 2.15,   ← (混淆的)
  "intv_EY_do_T_low": 0.31,      ← 干预后期望
  "intv_EY_do_T_high": 1.81,     ← (正确的因果量)
  "intv_ate": 1.50
}
```

**训练任务**:
- 输入: DAG + 观测条件概率 + do(T=t)
- 输出: E[Y | do(T=t)]
- 指标: MAE between predicted and true interventional means

---

## Type 4: 反事实推理 (Counterfactual)

**训练目标**: 给定已发生的事实，推理"如果当初做了不同选择会怎样"。

**数据格式**:

`counterfactuals.jsonl` (每行一个 JSON):
```json
{
  "problem_id": 0,
  "var_names": ["X0","X1","X2"],
  "edges": [["X0","X2"],["X1","X2"]],
  "observed": {"X0": 1.23, "X1": -0.45, "X2": 0.67},
  "intervention_var": "X1",
  "intervention_val": 2.0,
  "target": "X2",
  "counterfactual_val": 1.89
}
```

**三步反事实推理**:
1. Abduction (溯因): 从 observed 推断外生噪声 u
2. Action (行动): 施加 intervention
3. Prediction (预测): 计算 target 的反事实值

**训练任务**:
- 输入: (edges, observed, intervention)
- 输出: counterfactual_val 的估计值

---

## Type 5: 领域迁移 (Domain Transfer)

**训练目标**: 同一因果结构在不同语义领域的迁移推理。

**数据格式**:

`{domain}.csv` — 中文变量名:
```csv
年龄,治疗,康复
0.123,-0.456,0.789
...
```

`{domain}_en.csv` — 英文变量名:
```csv
Age,Treatment,Recovery
0.123,-0.456,0.789
...
```

`{domain}_meta.json`:
```json
{
  "domain": "medical",
  "var_names": ["Age","Treatment","Recovery"],
  "var_names_cn": ["年龄","治疗","康复"],
  "edges": [["Age","Treatment"],["Age","Recovery"],["Treatment","Recovery"]],
  "true_ate": 0.6,
  "obs_diff": 1.304,
  "confounding_bias": 0.704,
  "backdoor_set": ["Age"]
}
```

**8 个领域**: 医疗, 经济, 营销, 教育, 体育, 软件, 气候, 心理

**所有领域共享同一因果结构**: 混杂因子 → T → Y + 混杂因子 → Y
**真实 ATE 恒为 0.6**，但观测偏差从 0.700 到 0.797 变化。

**训练任务**:
- 输入: 领域描述 + 数据
- 输出: 识别因果结构 → 后门调整 → ATE ≈ 0.6
- 指标: 跨领域 ATE 估计的一致性

---

## 使用方法

```python
import json
import numpy as np

# 加载 Type 2 问题
with open("datasets/type2_effect_estimation/index.json") as f:
    index = json.load(f)

problem = index["problems"][0]
print(f"Treatment: {problem['treatment']}, Outcome: {problem['outcome']}")
print(f"True ATE: {problem['true_ate']}, Obs diff: {problem['obs_diff']}")

# 加载对应数据
data = np.genfromtxt(
    f"datasets/type2_effect_estimation/problem_{problem['problem_id']:04d}.csv",
    delimiter=',', skip_header=1
)
```

## 重新生成

```bash
/tmp/hydrogen_venv/bin/python generate_all_datasets.py
```

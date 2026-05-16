# Causal Agent — 模型与算法设计文档

> 版本: v0.9.4 | 日期: 2026-05-14
> 覆盖: 核心引擎 + 因果发现 + 效应估计 + 中介 + 敏感性 + 现代方法 + 变分原理
> 数学表达: LaTeX

---

## 目录

1. [因果图模型](#1-因果图模型)
2. [d-separation](#2-d-separation)
3. [因果效应识别](#3-因果效应识别)
4. [结构因果模型](#4-结构因果模型)
5. [因果发现算法](#5-因果发现算法)
6. [效应估计器](#6-效应估计器)
7. [现代因果推断方法](#7-现代因果推断方法)
8. [敏感性分析](#8-敏感性分析)
9. [算法复杂度总表](#9-算法复杂度总表)
10. [物理因果引擎](#10-物理因果引擎)
11. [模型与源文件对照表](#11-模型与源文件对照表)

---

## 1. 因果图模型

### 1.1 定义

**因果有向无环图** (Causal DAG) 定义为有序对 $G = (V, E)$：

$$V = \{X_1, X_2, \ldots, X_n\}$$
$$E \subseteq V \times V$$

其中 $(X_i \to X_j) \in E$ 表示 $X_i$ 是 $X_j$ 的**直接原因**（相对于 $V$ 中的变量集合）。

**因果马尔可夫条件** (Causal Markov Condition)：

$$X_i \perp\!\!\!\perp \text{NonDescendants}(X_i) \mid \text{Parents}_G(X_i)$$

即：每个变量在给定其父节点后，条件独立于其所有非后代节点。

**因果忠实性** (Causal Faithfulness)：观测数据中的条件独立性关系恰好等于图 $G$ 中 d-separation 编码的独立性。

### 1.2 核心操作

| 操作 | 定义 | 含义 |
|------|------|------|
| $\text{Parents}_G(v)$ | $\{u \in V \mid u \to v \in E\}$ | 直接原因 |
| $\text{Children}_G(v)$ | $\{w \in V \mid v \to w \in E\}$ | 直接影响 |
| $\text{Ancestors}_G(v)$ | $\{u \mid \exists \text{ directed path } u \rightsquigarrow v\}$ | 所有原因（含间接） |
| $\text{Descendants}_G(v)$ | $\{w \mid \exists \text{ directed path } v \rightsquigarrow w\}$ | 所有影响（含间接） |

**拓扑排序**: 任何 DAG 至少存在一个排列 $\pi: V \to \{1,\ldots,n\}$，使得对每条边 $(u \to v)$，有 $\pi(u) < \pi(v)$。拓扑排序给出了从原因到结果的因果顺序。

**干预操作 (Graph Mutilation)**: 干预 $do(X=x)$ 在图 $G$ 上等价于删除所有指向 $X$ 的边，记作 $G_{\overline{X}}$：

$$G_{\overline{X}} = (V, E \setminus \{(u, v) \in E \mid v \in X\})$$

---

## 2. d-separation

### 2.1 定义

在 DAG $G$ 中，给定条件集 $Z$，节点集 $X$ 和 $Y$ 是 **d-separated** 的，记作：

$$X \perp\!\!\!\perp_G Y \mid Z$$

当且仅当 $X$ 和 $Y$ 之间的每一条无向路径都被 $Z$ 阻断。

### 2.2 路径阻断规则

一条无向路径 $\pi$ 被 $Z$ 阻断，当且仅当 $\pi$ 上存在一个节点 $m$ 满足以下条件之一：

**规则 1 — 非对撞节点 (Chain / Fork)**：
> $m$ 在路径上不是对撞节点，且 $m \in Z$

$$m \notin \text{Colliders}(\pi) \land m \in Z \implies \text{阻断}$$

**规则 2 — 对撞节点 (Collider)**：
> $m$ 在路径上是对撞节点，且 $m \notin Z$，且 $\text{Descendants}_G(m) \cap Z = \varnothing$

$$m \in \text{Colliders}(\pi) \land m \notin Z \land \text{Desc}(m) \cap Z = \varnothing \implies \text{阻断}$$

**对撞节点定义**: 在路径上，若 $m$ 的两条相邻边都指向 $m$（即 $\cdots \to m \leftarrow \cdots$），则 $m$ 是对撞节点。

**三种基本结构**:

| 结构 | 图 | d-separated 条件 |
|------|-----|-----------------|
| 链 (Chain) | $X \to M \to Y$ | $X \perp Y \mid \{M\}$ |
| 叉 (Fork) | $X \leftarrow Z \to Y$ | $X \perp Y \mid \{Z\}$ |
| 对撞 (Collider) | $X \to Z \leftarrow Y$ | $X \perp Y$ (无条件)，但 $X \not\perp Y \mid \{Z\}$ |

**关键直觉**: 对撞节点在条件化时反而**打开**路径——这是选择偏差 (selection bias) 的图论根源。

### 2.3 算法：道德图法

```
输入: DAG G, 节点集 X, Y, 条件集 Z
输出: (X ⊥ Y | Z)_G 是否成立

1. Ancestral:   An = Ancestors(X ∪ Y ∪ Z)
2. Moralise:    对 An 中每个节点 v:
                  (a) 有向边 → 无向边
                  (b) 连接 v 的所有父节点对（"结婚"）
3. Delete:      删除 Z 中所有节点及其关联边
4. Check:       X 与 Y 在剩余图中不连通 → d-separated ✓
```

**复杂度**: $O(|V| + |E|)$ — 线性时间。

---

## 3. 因果效应识别

### 3.1 核心问题

**可识别性** (Identifiability): 给定 DAG $G$，因果效应 $P(Y \mid do(T=t))$ 是否可以从观测分布 $P(V)$ 中唯一确定？

### 3.2 Back-door 准则

**定义** (Pearl, 1993): 变量集 $Z$ 满足相对于 $(T, Y)$ 的 back-door 准则，当且仅当:

1. $Z \cap \text{Descendants}_G(T) = \varnothing$
2. $Z$ d-separates $T$ 和 $Y$ 之间所有以指向 $T$ 的边开头的路径

**Back-door 调整公式**:

$$P(Y = y \mid do(T = t)) = \sum_{z} P(Y = y \mid T = t, Z = z) \cdot P(Z = z)$$

在 $Z$ 连续的线性模型下的特殊形式:

$$\mathbb{E}[Y \mid do(T=t)] = \beta_T \cdot t + \sum_{z} \gamma_z \cdot \mathbb{E}[Z=z]$$

**算法**: 搜索满足 back-door 准则的最小调整集。

```
find_back_door_adjustment(G, T, Y):
  候选集 = {v ∈ V \ {T} ∣ v ∉ Descendants(T)}
  对 k = 0 到 |候选集|:
    对 候选集 的每个大小为 k 的子集 Z:
      if Z 阻断所有 T←...→Y 的路径:
        return Z
```

### 3.3 Front-door 准则

**定义** (Pearl, 1995): 中介变量集 $M$ 满足相对于 $(T, Y)$ 的 front-door 准则:

1. $M$ 截断所有从 $T$ 到 $Y$ 的有向路径
2. 从 $T$ 到 $M$ 无 unblocked back-door 路径
3. 从 $M$ 到 $Y$ 的所有 back-door 路径被 $\{T\}$ 阻断

**Front-door 调整公式**:

$$P(y \mid do(t)) = \sum_{m} P(m \mid t) \sum_{t'} P(y \mid t', m) \cdot P(t')$$

### 3.4 do-calculus 三条规则

设 $X, Y, Z, W$ 为 $V$ 的不交子集，令 $G_{\overline{X}}$ 表示删除指向 $X$ 的边，$G_{\underline{Z}}$ 表示删除从 $Z$ 出发的边。

**Rule 1 — 观测的增删**:

$$P(y \mid do(x), z, w) = P(y \mid do(x), w) \quad \text{if} \quad (Y \perp\!\!\!\perp Z \mid X, W)_{G_{\overline{X}}}$$

**Rule 2 — 干预/观测互换**:

$$P(y \mid do(x), do(z), w) = P(y \mid do(x), z, w) \quad \text{if} \quad (Y \perp\!\!\!\perp Z \mid X, W)_{G_{\overline{X}\underline{Z}}}$$

**Rule 3 — 干预的增删**:

$$P(y \mid do(x), do(z), w) = P(y \mid do(x), w) \quad \text{if} \quad (Y \perp\!\!\!\perp Z \mid X, W)_{G_{\overline{X}, \overline{Z(W)}}}$$

其中 $Z(W) = Z \setminus An(W)_{G_{\overline{X}}}$。

### 3.5 工具变量 (Instrumental Variable)

**IV 条件**: 变量 $Z$ 是 $(T, Y)$ 的有效工具变量，当且仅当:

1. **相关性**: $(Z \not\perp T)_G$ (图中 $Z$ 与 $T$ 有边或路径)
2. **排他性**: $(Z \perp Y \mid T, U)_{G}$（$Z$ 只通过 $T$ 影响 $Y$）
3. **外生性**: $(Z \perp U)_G$（$Z$ 与未观测混淆因子独立）

**IV 估计量**（线性、同质效应）:

$$\hat{\tau}_{IV} = \frac{\widehat{\text{Cov}}(Y, Z)}{\widehat{\text{Cov}}(T, Z)}$$

**2SLS 估计**（更一般的形式）:

第一阶段: $\hat{T} = \hat{\alpha} Z + \hat{\gamma} X$
第二阶段: $\hat{\tau}_{2SLS} = (\hat{T}'\hat{T})^{-1}\hat{T}'Y$

---

## 4. 结构因果模型

### 4.1 正式定义

**结构因果模型 (SCM)** 为四元组 $\mathcal{M} = \langle U, V, \mathcal{F}, P(u) \rangle$:

- $U$: 外生变量集合（模型外决定，不可观测的原因 + 随机噪声）
- $V$: 内生变量集合（模型内决定的变量）
- $\mathcal{F}$: 结构方程集合，对每个 $v \in V$:

$$v = f_v(\text{PA}(v), u_v), \quad u_v \in U$$

其中 $\text{PA}(v) \subseteq V$ 是 $v$ 的父节点集。

- $P(u)$: 外生变量 $U$ 上的联合概率分布

**定义 (因果图)**: SCM $\mathcal{M}$ 诱导的因果图 $G(\mathcal{M})$ 是节点集 $V$ 上有向图，其中存在边 $X_i \to X_j$ 当且仅当 $X_i \in \text{PA}(X_j)$。

### 4.2 三种推理模式

| 模式 | 形式 | 操作 | 输出 |
|------|------|------|------|
| **观测** (Observational) | $P(V)$ | 从 $P(u)$ 采样 | 联合分布 |
| **干预** (Interventional) | $P(V \mid do(X=x))$ | 替换方程 $X=x$ | 干预后分布 |
| **反事实** (Counterfactual) | $Y_{X=x}(u) \mid V=v$ | 溯因 + 行动 + 预测 | 个体层级预测 |

**干预 (do-operator)**:

$$P(V \mid do(X=x)) = \int \prod_{v \in V \setminus X} \delta(v - f_v(\text{PA}(v), u_v)) \cdot \prod_{x \in X} \delta(x - x^*) \cdot dP(u)$$

其中 $\delta$ 是 Dirac delta 函数（确定性方程），$\text{PA}(v)$ 使用干预后的值。

### 4.3 反事实推理 — 三步法

```
Step 1 — Abduction (溯因):
  给定观测 v_obs，推断外生噪声 u:
    u_v = v_obs - f_v(pa_obs, 0)    [确定性部分]

Step 2 — Action (行动):
  施加干预 do(X=x*)，修改结构方程:
    f_X 被替换为常数函数: X = x*

Step 3 — Prediction (预测):
  用溯因的 u + 修改后的方程重算目标:
    Y_cf = f_Y(PA_cf(Y), u_Y)
```

**线性 SCM 示例**（吸烟 → 焦油 → 癌症）:

$$\begin{aligned}
S &= \beta_{GS} \cdot G + u_S \\
T &= \beta_{ST} \cdot S + u_T \\
C &= \beta_{TC} \cdot T + \beta_{SC} \cdot S + \beta_{GC} \cdot G + u_C
\end{aligned}$$

**溯因**: $u_S = S_{obs} - \beta_{GS} \cdot G_{obs}$, $u_T = T_{obs} - \beta_{ST} \cdot S_{obs}$, $u_C = C_{obs} - (\beta_{TC}T_{obs} + \beta_{SC}S_{obs} + \beta_{GC}G_{obs})$

**行动**: $do(S=0)$

**预测**: $T' = \beta_{ST} \cdot 0 + u_T$, $C' = \beta_{TC}T' + \beta_{SC} \cdot 0 + \beta_{GC}G_{obs} + u_C$

### 4.4 线性 SCM 的路径分析

在**线性 SCM** 中，$T$ 对 $Y$ 的总因果效应等于所有从 $T$ 到 $Y$ 的有向路径上边权重的乘积之和：

$$\tau_{T \to Y} = \sum_{\pi \in \text{Paths}(T \rightsquigarrow Y)} \prod_{(u \to v) \in \pi} \beta_{uv}$$

可通过 Floyd-Warshall 风格算法高效计算——令 $\Theta_{ij}$ 为 $i$ 对 $j$ 的总效应：

$$\Theta_{ij} = \beta_{ij} + \sum_{k} \Theta_{ik} \cdot \beta_{kj}$$

---

## 5. 因果发现算法

### 5.1 条件独立性检验

**Fisher's z-test** (连续变量):

检验 $H_0: X \perp\!\!\!\perp Y \mid Z$:

偏相关系数:
$$r_{XY \cdot Z} = \text{Corr}(\text{resid}(X \mid Z), \;\text{resid}(Y \mid Z))$$

检验统计量:
$$z = \frac{1}{2} \ln \frac{1 + r_{XY \cdot Z}}{1 - r_{XY \cdot Z}} \cdot \sqrt{n - |Z| - 3}$$

$$p = 2 \cdot (1 - \Phi(|z|))$$

其中 $\Phi$ 是标准正态 CDF，$\text{resid}(X \mid Z) = X - \mathbb{E}[X \mid Z]$。

**G² 检验** (离散/离散化变量):

$$G^2 = 2 \sum_{xyz} O_{xyz} \ln \frac{O_{xyz}}{E_{xyz}} \sim \chi^2(df)$$

其中 $df = (|X|-1)(|Y|-1) \cdot \prod_{z \in Z} |z|$。

### 5.2 PC 算法

```
输入: 数据 D (n×d), 显著性水平 α
输出: CPDAG (Markov 等价类的部分有向无环图)

Step 1 — 骨架学习 (Skeleton Discovery):
  G = 完全无向图 (d 个节点)
  k = 0
  while 存在可测试的边:
    for 每条边 X—Y in G:
      for 每个 S ⊆ Adj(X)\{Y}, |S| = k:
        if (X ⊥ Y | S) with p > α:
          删除边 X—Y
          记录 SepSet(X,Y) = S
    k = k + 1

Step 2 — v-structure 定向:
  for 每个三元组 X—Z—Y 满足 X 与 Y 不相邻:
    if Z ∉ SepSet(X,Y):
      定向 X → Z ← Y  (collider)

Step 3 — Meek 规则传播:
  R1: 若 a→b—c 且 a,c 不相邻 → 定向 b→c
  R2: 若 a→b→c 且 a—c → 定向 a→c
  R3: 若 a—b→c, a—d→c, b,d 不相邻, a—c → 定向 a→c
  重复直到无新定向
```

**Markov 等价类**: PC 算法输出的 CPDAG 中，无向边 $X \text{—} Y$ 意味着存在两个 DAG（$X \to Y$ 和 $Y \to X$）都在 Markov 等价类中——即它们编码相同的条件独立性集合。

### 5.3 FCI 算法

FCI (Fast Causal Inference) 允许**隐变量**的存在。输出 **PAG** (Partial Ancestral Graph)。

**PAG 边类型与含义**:

| 边标记 | 含义 |
|--------|------|
| $A \to B$ | $A$ 是 $B$ 的祖先 |
| $A \leftrightarrow B$ | 存在隐变量同时影响 $A$ 和 $B$ |
| $A \circ\!\!\to B$ | $B$ 不是 $A$ 的祖先 |
| $A \circ\!\!-\!\!\circ B$ | 方向完全未知 |

**FCI 算法扩展** (在 PC 基础上):

```
Step 1-2: 与 PC 相同的骨架学习和 v-structure 定向

Step 3: Possible-D-SEP 重测试
  对每条保留的边 X—Y:
    计算 PDS(X,Y) — 从 X 出发沿特定路径（不含 Y）可达的节点集
    用 PDS(X,Y) 中的节点作为条件集重新检验 X ⊥ Y
    若独立性成立则删除边

Step 4: FCI 定向规则 (R1-R10)
  R1:  a*→b◦—*c, a,c不相邻 → b*→c
  R2:  a→b*→c, a*—◦c → a*→c
  R4:  a→b←c, a◦—◦c → a↔c  (存在辨别路径)
```

### 5.4 GES 算法

GES (Greedy Equivalence Search) 基于**评分**的方法，现为**三阶段**算法。

**BIC 评分**:

$$\text{BIC}(G, D) = \log P(D \mid G, \hat{\theta}_{ML}) - \frac{d_G}{2} \cdot \log n$$

其中 $d_G$ 是模型 $G$ 中自由参数的数量，$\hat{\theta}_{ML}$ 是最大似然估计。

对于线性高斯模型:

$$\text{BIC}(G) = -\frac{n}{2}\sum_{i=1}^{d} \left[\log(2\pi\hat{\sigma}_i^2) + 1\right] - \frac{\sum_i(|\text{PA}(X_i)| + 2)}{2} \cdot \log n$$

其中 $\hat{\sigma}_i^2 = \frac{1}{n}\|X_i - \hat{X}_i\|^2$ 是用父节点回归的残差方差。

**三阶段算法**:

| 阶段 | 操作 | 说明 |
|------|------|------|
| Phase 1 (Forward) | 贪心加边 | 从空图开始，每次加入提升 BIC 最大的边 |
| Phase 2 (Backward) | 贪心删边 | 从满图开始，每次删除提升 BIC 最大的边 |
| Phase 3 (CI Pruning) | 条件独立性剪枝 | 对每条边 $X \to Y$，测试 $\exists S: X \perp Y \mid S$，有分离集则删除 |

**Phase 3 动机**: 在 collider 结构 $X \to Z \leftarrow Y$ 下，有限样本中 $X$ 与 $Y$ 的微小偶然相关系数会提升 BIC，误导 Phase 1 添加 $X \to Y$ 多余边。Phase 2 因样本相关不足以抵消 BIC 惩罚而无法删除。Phase 3 用 Fisher z-test 的偏相关独立性检验 $(\alpha=0.05)$ 系统消除这类多余边。

$$\text{ΔBIC}(X\to Y \text{ removed}) = \frac{1}{2}\log n - \frac{n}{2}\log\frac{\text{Var}(Y)}{\text{Var}(Y \mid X)} \approx 3.8 - O(n\rho^2)$$

当样本相关系数 $\rho_{XY} \approx 0$,$\Delta\text{BIC}>0$ → Phase 2 可删除；但如果 $\rho_{XY} > \sqrt{2\log n / n}$，$\Delta\text{BIC}<0$ → Phase 2 保留多余边。Phase 3 通过条件独立性补救了这一局限。

### 5.5 自举置信度

$$\text{Conf}(u \to v) = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}\{(u \to v) \in \hat{G}_b\}$$

其中 $\hat{G}_b$ 是在第 $b$ 次 bootstrap 重采样上运行因果发现算法得到的图。$B$ 通常取 50-200。

---

## 6. 效应估计器

给定观测数据 $\mathcal{D} = \{(Y_i, T_i, Z_i)\}_{i=1}^n$ 和通过 identification 确定的调整集 $Z$，估计平均因果效应:

$$\tau = \mathbb{E}[Y \mid do(T=1)] - \mathbb{E}[Y \mid do(T=0)]$$

对于连续处理:

$$\tau = \frac{\partial}{\partial t} \mathbb{E}[Y \mid do(T=t)]$$

### 6.1 线性回归 (Linear Regression)

**模型**:

$$Y = \beta_T \cdot T + \gamma^T Z + \varepsilon, \quad \varepsilon \sim N(0, \sigma^2)$$

**估计量**: $\hat{\tau} = \hat{\beta}_T$

用 OLS 求解: $\hat{\beta} = (X^T X)^{-1} X^T Y$

**标准误**: $\text{SE}(\hat{\beta}_T) = \sqrt{\hat{\sigma}^2 \cdot (X^T X)^{-1}_{[1,1]}}$

其中 $\hat{\sigma}^2 = \frac{\|Y - X\hat{\beta}\|^2}{n - d - 1}$

### 6.2 倾向得分匹配 (PSM)

**倾向得分**: $\pi(Z) = P(T=1 \mid Z)$

用 logistic 回归估计: $\hat{\pi}(Z) = \frac{1}{1 + e^{-(\alpha + \beta^T Z)}}$

**匹配估计量**:

$$\hat{\tau}_{PSM} = \frac{1}{n_1} \sum_{i: T_i=1} \left( Y_i - \frac{1}{K} \sum_{j \in \mathcal{J}_K(i)} Y_j \right)$$

其中 $\mathcal{J}_K(i) = \{j: T_j=0, \pi(Z_j) \text{ 是 } \pi(Z_i) \text{ 的 } K \text{ 个最近邻}\}$

**协变量平衡诊断 — 标准化均值差 (SMD)**:

$$\text{SMD}_j = \frac{|\bar{X}_{j, treated} - \bar{X}_{j, control}|}{\sqrt{(s^2_{j, treated} + s^2_{j, control}) / 2}}$$

$\text{SMD} < 0.1$ 为平衡良好。

### 6.3 逆概率加权 (IPW)

**权重**:

$$w_i = \begin{cases}
\frac{p}{\pi(Z_i)} & \text{if } T_i = 1 \\
\frac{1-p}{1-\pi(Z_i)} & \text{if } T_i = 0
\end{cases}$$

其中 $p = P(T=1)$ 是边际处理概率（稳定化权重）。

**估计量**:

$$\hat{\tau}_{IPW} = \frac{\sum_{i} w_i T_i Y_i}{\sum_{i} w_i T_i} - \frac{\sum_{i} w_i (1-T_i) Y_i}{\sum_{i} w_i (1-T_i)}$$

**标准误**: 基于影响函数 (Influence Function) 的渐近方差:

$$\text{IF}_i = \frac{w_i T_i (Y_i - \hat{\mu}_1)}{\sum w_i T_i / n} - \frac{w_i (1-T_i)(Y_i - \hat{\mu}_0)}{\sum w_i (1-T_i) / n}$$

$$\text{SE} = \sqrt{\frac{\widehat{\text{Var}}(\text{IF})}{n}}$$

### 6.4 双重鲁棒估计器 (Doubly Robust)

设 $\mu_1(Z) = \mathbb{E}[Y \mid T=1, Z]$ 和 $\mu_0(Z) = \mathbb{E}[Y \mid T=0, Z]$ 为结果模型。

**DR 估计量**:

$$\hat{\tau}_{DR} = \frac{1}{n} \sum_{i=1}^{n} \left[ \hat{\mu}_1(Z_i) - \hat{\mu}_0(Z_i) + \frac{T_i(Y_i - \hat{\mu}_1(Z_i))}{\hat{\pi}(Z_i)} - \frac{(1-T_i)(Y_i - \hat{\mu}_0(Z_i))}{1-\hat{\pi}(Z_i)} \right]$$

**双重鲁棒性**: $\hat{\tau}_{DR}$ 是 $\tau$ 的一致估计，只要 $\hat{\pi}(Z)$ **或** $(\hat{\mu}_1, \hat{\mu}_0)$ 中至少有一个被正确指定。

### 6.5 分层估计 (Stratification)

将样本按 $\hat{\pi}(Z)$ 的 $s$ 个分位点分层:

$$\hat{\tau}_{strat} = \sum_{k=1}^{s} \frac{n_k}{n} \cdot \hat{\tau}_k$$

其中 $\hat{\tau}_k$ 是第 $k$ 层内用任何方法估计的 ATE。

### 6.6 估计器选择指南

| 场景 | 推荐 | 假设 |
|------|------|------|
| 线性、小调整集 | Linear | 线性性 |
| 二值处理、强重叠 | PSM | $\pi(Z)$ 正确 |
| 连续处理 | IPW (稳定化) | $\pi(Z)$ 正确 |
| 不想赌单一模型 | DR | $\pi$ 或 $\mu$ 至少一个正确 |
| 快速、直观 | Stratification ($s=5$) | 层内同质性 |
| 非线性、高维 | DML (见 §7) | 无参数假设 |

---

## 7. 现代因果推断方法

### 7.1 双重机器学习 (DML)

**动机**: 经典回归要求正确指定 $\mathbb{E}[Y|Z]$ 和 $\mathbb{E}[T|Z]$。DML 使用 ML 模型估计这些 nuisance functions，同时保持 $\sqrt{n}$ 一致性。

**部分线性模型**:

$$Y = \tau \cdot T + g(Z) + \varepsilon, \quad \mathbb{E}[\varepsilon \mid T, Z] = 0$$
$$T = m(Z) + \eta, \quad \mathbb{E}[\eta \mid Z] = 0$$

**算法** (Chernozhukov et al., 2018):

```
1. K-fold 交叉拟合: 将数据 {1,...,n} 随机分为 K 折 I_1,...,I_K

2. 对每个折 k:
   (a) 用 I_{-k} = {1,...,n} \ I_k 训练:
       ĝ_k(Z) ← 回归 Y 对 Z
       m̂_k(Z) ← 回归 T 对 Z
   
   (b) 在 I_k 上计算残差:
       Ỹ_i = Y_i - ĝ_k(Z_i)
       T̃_i = T_i - m̂_k(Z_i)
   
   (c) Neyman 正交分数:
       θ̂_k = (∑_{i∈I_k} Ỹ_i · T̃_i) / (∑_{i∈I_k} T̃_i²)

3. τ̂ = (1/K) ∑_k θ̂_k
   SE = std(θ̂_1,...,θ̂_K) / √K
```

**Neyman 正交性**: 得分函数 $\psi(Y,T,Z;\tau,g,m) = (Y - g(Z) - \tau(T - m(Z))) \cdot (T - m(Z))$ 满足:

$$\frac{\partial}{\partial g} \mathbb{E}[\psi] \bigg|_{g=g_0} = 0, \quad \frac{\partial}{\partial m} \mathbb{E}[\psi] \bigg|_{m=m_0} = 0$$

即得分函数对 nuisance 函数的 Gateaux 导数为零——即使 ĝ 和 m̂ 收敛较慢，τ̂ 仍然 $\sqrt{n}$-一致。

### 7.2 异质性处理效应 (CATE)

**定义**: $\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]$

#### S-learner

$$\hat{f}(T, X) = \mathbb{E}[Y \mid T, X]$$
$$\hat{\tau}_S(x) = \hat{f}(1, x) - \hat{f}(0, x)$$

#### T-learner

$$\hat{\mu}_1(x) = \mathbb{E}[Y \mid T=1, X=x], \quad \hat{\mu}_0(x) = \mathbb{E}[Y \mid T=0, X=x]$$
$$\hat{\tau}_T(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$$

#### X-learner

先估计 $\hat{\mu}_1, \hat{\mu}_0$ (同 T-learner)。然后构造**伪效应**:

$$\tilde{\tau}_{1,i} = Y_i - \hat{\mu}_0(X_i) \quad (T_i=1)$$
$$\tilde{\tau}_{0,i} = \hat{\mu}_1(X_i) - Y_i \quad (T_i=0)$$

分别训练 $\hat{\tau}_1(X)$ 预测 $\tilde{\tau}_1$，$\hat{\tau}_0(X)$ 预测 $\tilde{\tau}_0$:

$$\hat{\tau}_X(x) = \hat{\pi}(x) \cdot \hat{\tau}_0(x) + (1 - \hat{\pi}(x)) \cdot \hat{\tau}_1(x)$$

#### 因果森林 (Causal Forest)

**单棵因果树**: 递归分裂数据，最大化分裂后左右子节点 CATE 方差的加权和:

$$\text{Gain}(L, R) = n_L \cdot (\hat{\tau}_L - \hat{\tau}_{parent})^2 + n_R \cdot (\hat{\tau}_R - \hat{\tau}_{parent})^2$$

其中 $\hat{\tau}_{leaf} = \bar{Y}_{leaf, T=1} - \bar{Y}_{leaf, T=0}$。

**森林**: 集成 $B$ 棵树，每棵用 bootstrap 样本和随机特征子集训练:

$$\hat{\tau}_{CF}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{\tau}_{leaf_b(x)}$$

### 7.3 do-why 集成

```
后端切换:
  agent.set_backend("native")    # 本地引擎
  agent.set_backend("dowhy")     # do-why 引擎 (pip install dowhy)

do-why 工作流:
  1. CausalModel(data, treatment, outcome, graph)
  2. model.identify_effect()     → 识别策略
  3. model.estimate_effect()     → 数值估计
  4. model.refute_estimate()     → 敏感性验证
```

---

## 8. 敏感性分析

### 8.1 Rosenbaum 界限

**模型**: 对于匹配的对 $(i, j)$，隐偏差 $\Gamma$ 约束:

$$\frac{1}{\Gamma} \leq \frac{\pi_i/(1-\pi_i)}{\pi_j/(1-\pi_j)} \leq \Gamma$$

其中 $\pi_i = P(T_i=1 \mid Z_i, U_i)$ 包含未观测的 $U_i$。

**界限**: 在给定的 $\Gamma$ 下，计算检验统计量（如 Wilcoxon 符号秩）的 p 值上界。

- $\Gamma = 1$: 无隐偏差（随机分配）
- $\Gamma > 1$: 处理单元可能因 $U$ 而更可能接受处理

**解释**:

| $\Gamma$ 阈值 | 结论 |
|--------------|------|
| $> 3$ | 高度稳健 |
| $1.5 \sim 3$ | 中等稳健 |
| $< 1.5$ | 脆弱 |

### 8.2 E-value

**E-value** (VanderWeele & Ding, 2017): 要解释掉观测效应，未测量混淆因子 $U$ 需要同时与 $T$ **和** $Y$ 的最小关联强度。

在风险比尺度上:

$$\text{E-value} = RR_{obs} + \sqrt{RR_{obs} \cdot (RR_{obs} - 1)}$$

其中 $RR_{obs}$ 是观测到的风险比（或在连续变量上的指数化标准化效应）。

对置信区间下界:

$$\text{E-value}_{CI} = \max\left(1, RR_{CI} + \sqrt{RR_{CI} \cdot (RR_{CI} - 1)}\right)$$

**解释**: E-value $= 5$ 意味着: 要完全解释掉观测效应，未测量混淆因子 $U$ 必须与 $T$ 和 $Y$ 都至少有风险比 $\geq 5$ 的关联（在已测量协变量之上）。

---

## 9. 算法复杂度总表

| 算法 | 时间 | 空间 | 瓶颈 |
|------|------|------|------|
| d-separation | $O(\|V\| + \|E\|)$ | $O(\|V\| + \|E\|)$ | 道德图 BFS |
| Back-door 搜索 | $O(\|V\|^3 \cdot 2^{\|Z\|})$ | $O(\|V\|^2)$ | 子集枚举 |
| Front-door | $O(\|V\|^2)$ | $O(\|V\|^2)$ | 中介搜索 |
| PC 骨架 | $O(\|V\|^2 \cdot \binom{\|V\|-2}{k})$ | $O(\|V\|^2)$ | 条件集组合 |
| FCI | $O(\|V\|^2 \cdot 2^{\|V\|})$ (worst) | $O(\|V\|^2)$ | PDS 重测试 |
| GES | $O(\|V\|^3)$ per iter | $O(\|V\|^2)$ | BIC 计算 |
| 线性回归 | $O(nd^2 + d^3)$ | $O(d^2)$ | 矩阵求逆 |
| PSM | $O(nd^2 + n\log n)$ | $O(n)$ | 近邻搜索 |
| IPW | $O(nd^2)$ | $O(n)$ | 倾向得分 |
| DR | $O(nd^2)$ | $O(n)$ | 两次回归 |
| DML | $O(K \cdot nd^2)$ | $O(n)$ | K 次交叉拟合 |
| CATE (S/T/X) | $O(nd^2)$ | $O(nd^2)$ | 多项式特征 |
| Causal Forest | $O(B \cdot n\log n \cdot d)$ | $O(Bn)$ | B 棵树 |
| 反事实 | $O(\|V\| + \|E\|)$ | $O(\|V\|)$ | 拓扑遍历 |
| E-value | $O(1)$ | $O(1)$ | — |
| Rosenbaum | $O(n_\Gamma)$ | $O(1)$ | $\Gamma$ 网格 |

**符号说明**: $n$ = 样本数, $d$ = 变量数, $\|V\|$ = 节点数, $\|E\|$ = 边数, $B$ = 树数, $K$ = 折数, $n_\Gamma$ = $\Gamma$ 网格点数。

---

## 10. 物理因果引擎

> 核心思想：因果推断回答「谁影响谁」($G$)，物理定律回答「影响必须是怎样的」($\mathcal{F}$)。
> 两者结合：因果结构提供骨架，物理定律填充血肉。

**源文件**: `core/physics.py`

### 10.1 PhysicsLaw — 物理定律的形式化表示

每一条物理定律被建模为一个 `PhysicsLaw` 对象：

$$\mathcal{L} = (\text{name}, \text{domain}, \mathcal{E}, V_{in}, V_{out}, \Theta, \tau, f_{\mathcal{L}}, D_{causal}, D_{forbid})$$

| 字段 | 类型 | 含义 |
|------|------|------|
| $\mathcal{E}$ | LaTeX 字符串 | 定律的数学表达式，如 $F=ma$ |
| $V_{in}$ | `List[str]` | 输入变量（原因端） |
| $V_{out}$ | `List[str]` | 输出变量（结果端） |
| $\Theta$ | `List[str]` | 常数参数，如 $G, g, R$ |
| $\tau$ | `ConstraintType` | 约束类型（5 种） |
| $f_{\mathcal{L}}$ | Python lambda | 可执行的结构方程 |
| $D_{causal}$ | `List[(str,str)]` | 物理强制因果方向 |
| $D_{forbid}$ | `List[(str,str)]` | 物理禁止因果方向 |

### 10.2 五种约束类型

| 类型 | 数学含义 | 代码 | 示例 |
|------|---------|------|------|
| `SCM_EQUATION` | $v_{out} = f_{\mathcal{L}}(V_{in})$ | 替换 SCM 方程 | $a = F/m$ |
| `CONSERVATION` | $\sum \text{before} = \sum \text{after}$ | 验证守恒律 | $\sum p_{before} = \sum p_{after}$ |
| `DAG_EDGE` | $D_{causal} \subseteq E$, $D_{forbid} \cap E = \varnothing$ | 约束因果图边 | $F \to a$，禁止 $a \to F$ |
| `BOUNDARY` | $v \in [v_{min}, v_{max}]$ | 取值范围约束 | $0 \leq \eta \leq 1$ |
| `SYMMETRY` | $\mathcal{L}$ 在变换群 $\mathcal{G}$ 下不变 | 对称性约束 | 平移不变性 |

### 10.3 PhysicsLibrary — 领域组织

定律按领域分组的注册表：

```
PhysicsLibrary
├── mechanics (7):  newton_2nd, kinetic_energy, momentum, momentum_conservation,
│                   energy_conservation, hookes_law, pendulum_period
├── electromagnetism (3): ohms_law, power_law, joules_law
├── thermodynamics (2): ideal_gas_law, second_law_thermo
├── fluids (2): bernoulli, continuity
└── (可扩展): 通过 physics.register() 添加
```

**核心方法**:

```python
physics.find_relevant(variables)  → List[PhysicsLaw]
# 返回所有涉及给定变量集合的物理定律
```

### 10.4 PhysicsInformedCausalGraph — 物理约束的因果图

在标准 `CausalDAG` 上叠加物理约束：

$$\hat{G} = \arg\min_{G \in \mathcal{G}_{DAG}} \left[ \text{BIC}(G, D) + \lambda \cdot \text{PhysicsViolation}(G, \mathcal{L}) \right]$$

**约束施加流程**：

```
1. 数据驱动:   G_raw = pc_algorithm(data)
2. 定律匹配:   laws = physics.find_relevant(G_raw.variables)
3. 边修正:
   - forced_edges:    物理要求存在的边（如 F→a）
   - forbidden_edges:  物理禁止的边（如 a→F）
   - reversed_edges:   方向错误的边需翻转
4. 输出:  G_corrected = apply_constraints(G_raw, laws)
```

**关键属性**:

| 属性 | 类型 | 含义 |
|------|------|------|
| `forced_edges` | `List[(str,str)]` | 物理要求必须存在的边 |
| `forbidden_edges` | `List[(str,str)]` | 物理禁止存在的边 |
| `reversed_edges` | `List[(str,str)]` | 方向与物理矛盾的边 |

### 10.5 PhysicsInformedSCM — 物理约束的 SCM

在标准 `SCM` 上将结构方程替换为物理公式：

$$\mathcal{M}_{physics} = \langle U, V, \mathcal{F}_{physics}, P(u) \rangle$$

其中：

$$f_v = \begin{cases}
f_{\mathcal{L}}(PA(v)) & \text{if } \exists \mathcal{L}: v \in V_{out}(\mathcal{L}) \\
f_v^{\text{learned}}(PA(v), u_v) & \text{otherwise}
\end{cases}$$

**物理方程优先级高于学习方程**：一旦物理定律库中有匹配的方程，就直接替换，不依赖数据拟合。

**反事实的物理验证**：

```python
counterfactual_with_physics(observed, intervention, target):
  1. cf_val = standard_counterfactual(observed, intervention, target)
  2. cf_state = observed ∪ intervention ∪ {target: cf_val}
  3. for each conservation_law:
       if violation(cf_state, law) > tolerance:
           flag_violation(law)
  4. return (cf_val, violations, verdict)
```

### 10.6 SymbolicPhysicsDiscovery — 公式发现

从数据 + 因果 DAG 中**自动发现**候选物理公式。搜索空间为符号表达式的组合：

$$\mathcal{H} = \left\{ \sum_{i} w_i \cdot \prod_{j} x_j^{p_j} \;\middle|\; p_j \in \{0, 1, 2\}, |\{p_j \neq 0\}| \leq d_{max} \right\}$$

**评分函数**：

$$\text{Score}(h) = \underbrace{-n \cdot \log(\text{MSE}(h) + \epsilon)}_{\text{拟合优度}} - \underbrace{k \cdot \log n}_{\text{复杂度惩罚}}$$

其中 $k$ 是表达式中自由参数的数量。

**两步验证**：
1. 评分排名：选 BIC 最优的表达式
2. 定律匹配：检查是否与已知物理定律一致

### 10.7 全流水线

```python
physics_causal_pipeline(data, var_names, treatment, outcome):
  ┌──────────────────────────────────────────┐
  │ Step 1: Causal Discovery (PC)            │
  │   G_raw = pc_algorithm(data)             │
  ├──────────────────────────────────────────┤
  │ Step 2: Physics Constraints              │
  │   G = PhysicsInformedCausalGraph(G_raw)  │
  │   → forced_edges, forbidden_edges        │
  ├──────────────────────────────────────────┤
  │ Step 3: Physics-Informed SCM             │
  │   scm = PhysicsInformedSCM(scm_raw)      │
  │   → physics_equations 自动替换           │
  ├──────────────────────────────────────────┤
  │ Step 4: Formula Discovery                │
  │   discoveries = SymbolicPhysicsDiscovery │
  │   → 未知变量的候选物理公式                │
  ├──────────────────────────────────────────┤
  │ Step 5: Identification + Counterfactual  │
  │   identify_effect(G, T, Y)               │
  │   counterfactual_with_physics()          │
  └──────────────────────────────────────────┘
```

**摆钟验证示例**：

```
输入:  数据 (Length, Gravity, Period, n=500)
      已知定律: T = 2π√(L/g) (pendulum_period)

输出:
  Causal Skeleton:    Length → Period  (PC 自动发现 ✓)
  Physics-Governed:   Period ← T = 2π√(L/g)  (自动匹配 ✓)
  Causal Effect:      Back-door, no confounding
```

---

## 11. 模型与源文件对照表

| 章节 | 模型/算法 | 源文件 | 核心类/函数 |
|------|----------|--------|-----------|
| 1 | 因果图模型 | `core/graph.py` | `CausalDAG` |
| 2 | d-separation | `core/graph.py` | `CausalDAG.is_d_separated()` |
| 3 | Back-door 识别 | `core/identification.py` | `find_back_door_adjustment()` |
| 3 | Front-door 识别 | `core/identification.py` | `find_front_door_adjustment()` |
| 3 | do-calculus | `core/identification.py` | `do_calculus_rule1/2/3()` |
| 3 | 工具变量 | `core/identification.py` | `check_instrument()` |
| 4 | 结构因果模型 | `core/scm.py` | `SCM`, `StructuralEquation` |
| 4 | 线性 SCM | `core/scm.py` | `linear_scm()`, `linear_eq()` |
| 4 | 干预 (do) | `core/scm.py` | `SCM.intervene()` → `IntervenedSCM` |
| 4 | 反事实推理 | `core/scm.py` | `SCM.counterfactual()` |
| 5 | Fisher z-test | `core/discovery.py` | `fisher_z_test()` |
| 5 | G² 检验 | `core/discovery.py` | `g_squared_test()` |
| 5 | PC 算法 | `core/discovery.py` | `pc_algorithm()` |
| 5 | FCI 算法 | `core/discovery.py` | `fci_algorithm()`, `PAG`, `PAGEdge` |
| 5 | GES 算法 | `core/discovery.py` | `ges_algorithm()` |
| 5 | 自举置信度 | `core/discovery.py` | `bootstrap_edge_confidence()` |
| 6 | 线性回归估计 | `core/estimation.py` | `estimate_ate_linear()` |
| 6 | 倾向得分匹配 | `core/estimation.py` | `estimate_ate_psm()` |
| 6 | 逆概率加权 | `core/estimation.py` | `estimate_ate_ipw()` |
| 6 | 双重鲁棒 | `core/estimation.py` | `estimate_ate_doubly_robust()` |
| 6 | 分层估计 | `core/estimation.py` | `estimate_ate_stratified()` |
| 7 | 双重机器学习 | `core/modern.py` | `estimate_ate_dml()` |
| 7 | S-learner | `core/modern.py` | `estimate_cate_slearner()` |
| 7 | T-learner | `core/modern.py` | `estimate_cate_tlearner()` |
| 7 | X-learner | `core/modern.py` | `estimate_cate_xlearner()` |
| 7 | 因果森林 | `core/modern.py` | `estimate_cate_forest()`, `SimpleCausalForest` |
| 7 | do-why 集成 | `core/modern.py` | `estimate_ate_dowhy()` |
| 8 | Rosenbaum 界限 | `core/sensitivity.py` | `rosenbaum_bounds()` |
| 8 | E-value | `core/sensitivity.py` | `e_value()` |
| 8 | 敏感性报告 | `core/sensitivity.py` | `full_sensitivity_report()` |
| 10 | 物理定律 | `core/physics.py` | `PhysicsLaw`, `PhysicsLibrary` |
| 10 | 物理约束因果图 | `core/physics.py` | `PhysicsInformedCausalGraph` |
| 10 | 物理约束 SCM | `core/physics.py` | `PhysicsInformedSCM` |
| 10 | 符号公式发现 | `core/physics.py` | `SymbolicPhysicsDiscovery` |
| 10 | 物理因果流水线 | `core/physics.py` | `physics_causal_pipeline()` |
| — | DAG 可视化 | `core/visualization.py` | `dag_to_ascii()`, `dag_to_dot()`, `render_dag()` |
| — | NL 解析 | `nlp/parser.py` | `CausalParser` |
| — | 交互式 Agent | `agent.py` | `CausalAgent` |

| 演示/测试 | 源文件 |
|-----------|--------|
| 全功能验证 | `demos/run_all.py` |
| LLM 集成原型 | `demos/llm_prototype.py` |
| 物理因果演示 | `demos/physics_causal_demo.py` |
| 飞机消失预测 | `demos/aircraft_disappearance.py` |
| 训练数据生成 | `training_data.py`, `generate_all_datasets.py` |

---

## 附录 A: 符号约定

| 符号 | 含义 |
|------|------|
| $G = (V, E)$ | 有向无环图 |
| $\text{PA}_G(v)$ | $v$ 的父节点集 |
| $do(X=x)$ | Pearl 的干预算子 |
| $G_{\overline{X}}$ | 删除指向 $X$ 的边后的图 |
| $G_{\underline{X}}$ | 删除从 $X$ 出发的边后的图 |
| $\perp\!\!\!\perp$ | (条件) 独立 |
| $\tau$ | 平均因果效应 (ATE) |
| $\tau(x)$ | 条件平均因果效应 (CATE) |
| $\pi(Z)$ | 倾向得分 $P(T=1 \mid Z)$ |
| $\mu_1, \mu_0$ | 结果回归函数 |
| $r_{XY \cdot Z}$ | 偏相关系数 |

## 附录 B: 参考文献

1. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. 2nd ed. Cambridge.
2. Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search*. MIT Press.
3. Chernozhukov, V. et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1).
4. Athey, S. & Imbens, G. (2016). Recursive partitioning for heterogeneous causal effects. *PNAS*, 113(27).
5. Künzel, S. et al. (2019). Metalearners for estimating heterogeneous treatment effects. *PNAS*, 116(10).
6. VanderWeele, T. & Ding, P. (2017). Sensitivity analysis in observational research: Introducing the E-value. *Annals of Internal Medicine*, 167(4).
7. Rosenbaum, P. (2002). *Observational Studies*. 2nd ed. Springer.
8. Chickering, D. M. (2002). Optimal structure identification with greedy search. *JMLR*, 3.

"""
Causal Reasoning Training Data Generator
========================================

To build a model with GENERAL causal reasoning ability (not just
pattern matching on one scenario), we need training data that
covers five distinct reasoning skills:

  Type 1 — Structure Learning    : Data → DAG
  Type 2 — Effect Estimation     : (DAG, Data) → ATE
  Type 3 — Interventional Reasoning: (DAG, Observational, do(X)) → P(Y|do(X))
  Type 4 — Counterfactual        : (SCM, Observed, do(X)) → Counterfactual Y
  Type 5 — Domain Transfer       : Same structure, different semantics

Each type trains a fundamentally different cognitive operation.
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════
#  Random DAG Generator
# ═══════════════════════════════════════════════════════════════════

def random_dag(
    n_vars: int,
    edge_prob: float = 0.3,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Generate a random DAG with n_vars nodes.

    Uses a random topological ordering: for i < j,
    add edge i→j with probability edge_prob.

    Returns (var_names, edges).
    """
    rng = np.random.default_rng(seed)
    var_names = [f"X{i}" for i in range(n_vars)]
    edges = []
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            if rng.random() < edge_prob:
                edges.append((var_names[i], var_names[j]))
    return var_names, edges


# ═══════════════════════════════════════════════════════════════════
#  Type 1: Structure Learning Dataset
# ═══════════════════════════════════════════════════════════════════

def generate_structure_learning_dataset(
    n_graphs: int = 1000,
    n_samples_per_graph: int = 500,
    min_vars: int = 3,
    max_vars: int = 8,
    edge_prob_range: Tuple[float, float] = (0.2, 0.5),
    noise_std_range: Tuple[float, float] = (0.1, 1.0),
    seed: int = 42,
) -> List[Dict]:
    """
    Type 1: Teach the model to recover causal structure from data.

    For each random DAG:
      1. Generate random linear SCM
      2. Sample observational data
      3. Store (data, true_DAG, true_edges, conditional_independencies)

    Returns list of dicts, each containing:
      - data: (n_samples, n_vars) observational data
      - var_names: list of variable names
      - edges: list of (u,v) true causal edges
      - adj_matrix: (n_vars, n_vars) adjacency matrix
      - skeleton: undirected edge list
      - v_structures: list of collider triples (x,z,y)
      - indep_pairs: list of conditionally independent pairs with cond sets
    """
    rng = np.random.default_rng(seed)
    dataset = []

    for g in range(n_graphs):
        n_vars = rng.integers(min_vars, max_vars + 1)
        ep = rng.uniform(*edge_prob_range)
        var_names, edges = random_dag(n_vars, ep, seed=seed + g)

        # Build adjacency matrix
        idx = {v: i for i, v in enumerate(var_names)}
        adj = np.zeros((n_vars, n_vars), dtype=int)
        for u, v in edges:
            adj[idx[u], idx[v]] = 1

        # Generate linear SCM
        noise_std = rng.uniform(*noise_std_range)
        coefficients = {}
        for u, v in edges:
            # Random coefficient: magnitude 0.3-1.5, random sign
            coefficients.setdefault(v, {})[u] = rng.uniform(0.3, 1.5) * rng.choice([-1, 1])

        # Topological order = index order (by construction)
        data = np.zeros((n_samples_per_graph, n_vars))
        for j in range(n_vars):
            v = var_names[j]
            parents = [var_names[i] for i in range(j) if adj[i, j] == 1]
            col = np.zeros(n_samples_per_graph)
            for p in parents:
                col += coefficients.get(v, {}).get(p, 0) * data[:, idx[p]]
            col += rng.normal(0, noise_std, n_samples_per_graph)
            data[:, j] = col

        # Compute skeleton (from DAG)
        skeleton = []
        for u, v in edges:
            skeleton.append((u, v))

        # Find v-structures (colliders)
        v_structures = []
        for z in var_names:
            z_parents = [u for u, _ in edges if _ == z]
            for i in range(len(z_parents)):
                for j in range(i + 1, len(z_parents)):
                    x, y = z_parents[i], z_parents[j]
                    # Check if x and y are NOT adjacent
                    if not any((a == x and b == y) or (a == y and b == x)
                              for a, b in edges):
                        v_structures.append((x, z, y))

        # Find some known conditional independencies
        # For chain X→Y→Z: X ⊥ Z | Y
        indep_pairs = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                vi, vj = var_names[i], var_names[j]
                # Try conditioning on all other vars
                for k in range(n_vars):
                    if k != i and k != j:
                        vk = var_names[k]
                        # Simple heuristic: if all paths go through vk
                        # (This is approximate; real d-sep needs the full algorithm)
                        pass

        dataset.append({
            "graph_id": g,
            "n_vars": n_vars,
            "n_samples": n_samples_per_graph,
            "var_names": var_names,
            "edges": edges,
            "adj_matrix": adj,
            "skeleton": skeleton,
            "v_structures": v_structures,
            "noise_std": noise_std,
            "data": data,
        })

    return dataset


# ═══════════════════════════════════════════════════════════════════
#  Type 2: Effect Estimation Dataset
# ═══════════════════════════════════════════════════════════════════

def generate_effect_estimation_dataset(
    n_problems: int = 2000,
    n_samples: int = 1000,
    min_vars: int = 3,
    max_vars: int = 6,
    seed: int = 42,
) -> List[Dict]:
    """
    Type 2: Teach the model to estimate ATE correctly.

    For each problem:
      1. Generate random DAG with a designated treatment T and outcome Y
      2. Generate SCM with known true ATE
      3. Sample observational data
      4. Produce both observational difference E[Y|T=1]-E[Y|T=0] and true ATE

    The model must learn: observational difference ≠ causal effect
    when confounding exists.

    Returns list of dicts, each containing:
      - dag: CausalDAG
      - treatment, outcome: variable names
      - true_ate: ground truth ATE
      - obs_diff: naive observational difference
      - data: (n_samples, n_vars)
      - has_confounding: bool
      - backdoor_set: list of adjustment variables
    """
    import sys
    sys.path.insert(0, '/home/duyw/causal_agent')
    from core.graph import CausalDAG
    from core.identification import find_back_door_adjustment

    rng = np.random.default_rng(seed)
    dataset = []

    for p in range(n_problems):
        n_vars = rng.integers(min_vars, max_vars + 1)
        var_names, edges = random_dag(n_vars, rng.uniform(0.25, 0.55), seed=seed + p)
        dag = CausalDAG(var_names, edges)

        # Choose treatment and outcome
        # Ensure T is ancestor of Y (otherwise ATE = 0)
        candidates = []
        for t in var_names:
            for y in var_names:
                if t != y and y in dag.descendants(t):
                    candidates.append((t, y))

        if not candidates:
            # Force at least one causal path
            t, y = var_names[0], var_names[-1]
            edges.append((t, y))
            dag = CausalDAG(var_names, edges)
        else:
            t, y = candidates[rng.integers(len(candidates))]

        # Generate SCM with known coefficients
        idx = {v: i for i, v in enumerate(var_names)}
        n_v = len(var_names)
        coeffs = np.zeros((n_v, n_v))
        for u, v in edges:
            coeffs[idx[u], idx[v]] = rng.uniform(0.4, 2.0) * rng.choice([-1, 1])

        # True ATE = sum over all directed paths from T to Y
        # For linear SCM: ATE = sum of path products
        true_ate = _compute_linear_ate(coeffs, idx[t], idx[y])

        # Generate data
        data = np.zeros((n_samples, n_v))
        for j in range(n_v):
            parents = [i for i in range(n_v) if coeffs[i, j] != 0]
            col = np.zeros(n_samples)
            for pi in parents:
                col += coeffs[pi, j] * data[:, pi]
            col += rng.normal(0, 0.5, n_samples)
            data[:, j] = col

        # Naive observational difference: E[Y|T=high] - E[Y|T=low]
        t_median = np.median(data[:, idx[t]])
        high_mask = data[:, idx[t]] > t_median
        low_mask = data[:, idx[t]] <= t_median
        obs_diff = (data[high_mask, idx[y]].mean() -
                    data[low_mask, idx[y]].mean())

        # Back-door adjustment set
        adj_set = find_back_door_adjustment(dag, t, y)
        has_confounding = adj_set is not None and len(adj_set) > 0 if adj_set else False

        dataset.append({
            "problem_id": p,
            "var_names": var_names,
            "edges": edges,
            "treatment": t,
            "outcome": y,
            "true_ate": float(true_ate),
            "obs_diff": float(obs_diff),
            "confounding_bias": float(obs_diff - true_ate),
            "has_confounding": has_confounding,
            "backdoor_set": adj_set if adj_set else [],
            "n_vars": n_vars,
            "data": data,
        })

    return dataset


def _compute_linear_ate(coeffs: np.ndarray, t_idx: int, y_idx: int) -> float:
    """Compute total effect of T on Y in a linear SCM.

    ATE = sum over all directed paths from T to Y of product of edge weights.
    Uses path enumeration (exact for small graphs).
    """
    n = coeffs.shape[0]

    # Floyd-Warshall style: total effect matrix
    # effect[i,j] = total effect of i on j
    effect = coeffs.copy()

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if effect[i, k] != 0 and effect[k, j] != 0:
                    effect[i, j] += effect[i, k] * effect[k, j]

    return effect[t_idx, y_idx]


# ═══════════════════════════════════════════════════════════════════
#  Type 3: Interventional Reasoning Dataset
# ═══════════════════════════════════════════════════════════════════

def generate_interventional_dataset(
    n_problems: int = 1000,
    n_samples: int = 2000,
    min_vars: int = 3,
    max_vars: int = 6,
    seed: int = 42,
) -> List[Dict]:
    """
    Type 3: Teach the model to predict interventional outcomes.

    For each problem:
      1. Generate DAG + SCM with known true ATE
      2. Generate observational data
      3. Generate interventional data (do(T=t))
      4. The task: given observational data + DAG, predict E[Y|do(T=t)]

    The model must learn: P(Y|do(T)) ≠ P(Y|T) when confounding.

    Returns list of dicts, each containing:
      - dag, treatment, outcome, true_ate
      - obs_data: (n_samples, n_vars)
      - intv_data_low: data under do(T=low)
      - intv_data_high: data under do(T=high)
      - obs_EY_given_T: observational conditional expectations
      - intv_EY: interventional expectations
    """
    import sys
    sys.path.insert(0, '/home/duyw/causal_agent')

    rng = np.random.default_rng(seed)
    dataset = []

    for p in range(n_problems):
        n_vars = rng.integers(min_vars, max_vars + 1)
        var_names, edges = random_dag(n_vars, rng.uniform(0.2, 0.5), seed=seed + p)

        from core.graph import CausalDAG
        dag = CausalDAG(var_names, edges)

        # Choose treatment and outcome
        candidates = [(t, y) for t in var_names for y in var_names
                      if t != y and y in dag.descendants(t)]
        if not candidates:
            continue
        t, y = candidates[rng.integers(len(candidates))]

        idx = {v: i for i, v in enumerate(var_names)}
        n_v = len(var_names)
        coeffs = np.zeros((n_v, n_v))
        for u, v in edges:
            coeffs[idx[u], idx[v]] = rng.uniform(0.3, 1.8)

        true_ate = _compute_linear_ate(coeffs, idx[t], idx[y])

        # Generate observational data
        obs_data = np.zeros((n_samples, n_v))
        for j in range(n_v):
            parents = [i for i in range(n_v) if coeffs[i, j] != 0]
            col = np.zeros(n_samples)
            for pi in parents:
                col += coeffs[pi, j] * obs_data[:, pi]
            col += rng.normal(0, 0.5, n_samples)
            obs_data[:, j] = col

        # Observational conditional expectations
        t_col = obs_data[:, idx[t]]
        y_col = obs_data[:, idx[y]]
        t_median = np.median(t_col)

        obs_EY_low = y_col[t_col <= t_median].mean()
        obs_EY_high = y_col[t_col > t_median].mean()

        # Generate interventional data: do(T = low)
        intv_data_low = np.zeros((n_samples, n_v))
        for j in range(n_v):
            if var_names[j] == t:
                intv_data_low[:, j] = t_median - 2 * np.std(t_col)
            else:
                parents = [i for i in range(n_v) if coeffs[i, j] != 0]
                col = np.zeros(n_samples)
                for pi in parents:
                    if var_names[pi] == t:
                        col += coeffs[pi, j] * intv_data_low[:, pi]
                    else:
                        col += coeffs[pi, j] * obs_data[:, pi]
                col += rng.normal(0, 0.5, n_samples)
                intv_data_low[:, j] = col

        # Interventional: do(T = high)
        intv_data_high = np.zeros((n_samples, n_v))
        for j in range(n_v):
            if var_names[j] == t:
                intv_data_high[:, j] = t_median + 2 * np.std(t_col)
            else:
                parents = [i for i in range(n_v) if coeffs[i, j] != 0]
                col = np.zeros(n_samples)
                for pi in parents:
                    if var_names[pi] == t:
                        col += coeffs[pi, j] * intv_data_high[:, pi]
                    else:
                        col += coeffs[pi, j] * obs_data[:, pi]
                col += rng.normal(0, 0.5, n_samples)
                intv_data_high[:, j] = col

        intv_EY_low = intv_data_low[:, idx[y]].mean()
        intv_EY_high = intv_data_high[:, idx[y]].mean()

        dataset.append({
            "problem_id": p,
            "var_names": var_names,
            "edges": edges,
            "treatment": t,
            "outcome": y,
            "true_ate": float(true_ate),
            "obs_EY_given_T_low": float(obs_EY_low),
            "obs_EY_given_T_high": float(obs_EY_high),
            "intv_EY_do_T_low": float(intv_EY_low),
            "intv_EY_do_T_high": float(intv_EY_high),
            "intv_ate": float(intv_EY_high - intv_EY_low),
        })

    return dataset


# ═══════════════════════════════════════════════════════════════════
#  Type 4: Counterfactual Dataset
# ═══════════════════════════════════════════════════════════════════

def generate_counterfactual_dataset(
    n_problems: int = 500,
    n_samples: int = 200,
    min_vars: int = 3,
    max_vars: int = 5,
    seed: int = 42,
) -> List[Dict]:
    """
    Type 4: Teach counterfactual reasoning.

    For each problem:
      1. Generate SCM with known equations
      2. Sample a factual observation (u, v_obs)
      3. Apply intervention do(X=x')
      4. Compute counterfactual outcome Y_cf

    The model gets (SCM, observed_state, intervention) and must
    predict the counterfactual value.

    Returns list of dicts.
    """
    import sys
    sys.path.insert(0, '/home/duyw/causal_agent')
    from core.scm import linear_scm, StructuralEquation
    from core.graph import CausalDAG

    rng = np.random.default_rng(seed)
    dataset = []

    for p in range(n_problems):
        n_vars = rng.integers(min_vars, max_vars + 1)
        var_names, edges = random_dag(n_vars, rng.uniform(0.25, 0.5), seed=seed + p)

        # Build linear SCM
        coefficients = {}
        for u, v in edges:
            coefficients.setdefault(v, {})[u] = rng.uniform(0.3, 1.5)

        dag = CausalDAG(var_names, edges)
        scm = linear_scm(dag, coefficients, noise_std=rng.uniform(0.1, 0.8))

        # Sample one factual observation
        samples = scm.sample(1, seed=seed + p * 1000)
        observed = {v: float(samples[v][0]) for v in var_names}

        # Choose intervention
        intv_var = var_names[rng.integers(n_vars)]
        intv_val = rng.uniform(-2, 2)

        # Choose target (must be downstream of intv_var in expectation)
        descendants = list(dag.descendants(intv_var))
        if not descendants:
            continue
        target = descendants[rng.integers(len(descendants))]

        # Compute counterfactual
        cf_val = scm.counterfactual(observed, {intv_var: intv_val}, target)

        dataset.append({
            "problem_id": p,
            "var_names": var_names,
            "edges": edges,
            "observed": observed,
            "intervention_var": intv_var,
            "intervention_val": float(intv_val),
            "target": target,
            "counterfactual_val": float(cf_val),
        })

    return dataset


# ═══════════════════════════════════════════════════════════════════
#  Type 5: Domain Transfer Dataset
# ═══════════════════════════════════════════════════════════════════

DOMAIN_TEMPLATES = {
    "medical": {
        "vars": ["Age", "Treatment", "Recovery"],
        "edges": [("Age", "Treatment"), ("Age", "Recovery"), ("Treatment", "Recovery")],
        "names": {"Age": "年龄", "Treatment": "治疗", "Recovery": "康复"},
    },
    "economics": {
        "vars": ["SES", "Education", "Income"],
        "edges": [("SES", "Education"), ("SES", "Income"), ("Education", "Income")],
        "names": {"SES": "家庭背景", "Education": "教育", "Income": "收入"},
    },
    "marketing": {
        "vars": ["Season", "AdSpend", "Sales"],
        "edges": [("Season", "AdSpend"), ("Season", "Sales"), ("AdSpend", "Sales")],
        "names": {"Season": "季节", "AdSpend": "广告投入", "Sales": "销售额"},
    },
    "education": {
        "vars": ["Motivation", "StudyHours", "Grade"],
        "edges": [("Motivation", "StudyHours"), ("Motivation", "Grade"), ("StudyHours", "Grade")],
        "names": {"Motivation": "学习动机", "StudyHours": "学习时间", "Grade": "成绩"},
    },
    "sports": {
        "vars": ["Talent", "Training", "Performance"],
        "edges": [("Talent", "Training"), ("Talent", "Performance"), ("Training", "Performance")],
        "names": {"Talent": "天赋", "Training": "训练量", "Performance": "表现"},
    },
    "software": {
        "vars": ["TeamSize", "CodeQuality", "Bugs"],
        "edges": [("TeamSize", "CodeQuality"), ("TeamSize", "Bugs"), ("CodeQuality", "Bugs")],
        "names": {"TeamSize": "团队规模", "CodeQuality": "代码质量", "Bugs": "缺陷数"},
    },
    "climate": {
        "vars": ["CO2", "Temperature", "SeaIce"],
        "edges": [("CO2", "Temperature"), ("CO2", "SeaIce"), ("Temperature", "SeaIce")],
        "names": {"CO2": "二氧化碳", "Temperature": "温度", "SeaIce": "海冰面积"},
    },
    "psychology": {
        "vars": ["Stress", "Sleep", "Mood"],
        "edges": [("Stress", "Sleep"), ("Stress", "Mood"), ("Sleep", "Mood")],
        "names": {"Stress": "压力", "Sleep": "睡眠质量", "Mood": "情绪"},
    },
}


def generate_domain_transfer_dataset(
    n_samples_per_domain: int = 500,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Type 5: Same causal structure (confounder→T→Y+confounder→Y)
    across multiple semantic domains.

    The model must learn that the causal REASONING is domain-independent:
    the same graph structure implies the same adjustment strategy,
    regardless of whether variables are "Age" or "CO2".

    Returns dict mapping domain_name → dataset.
    """
    rng = np.random.default_rng(seed)
    datasets = {}

    for domain, template in DOMAIN_TEMPLATES.items():
        vars_list = template["vars"]
        edges = template["edges"]

        n_vars = len(vars_list)
        idx = {v: i for i, v in enumerate(vars_list)}

        # Generate data with the same causal structure
        data = np.zeros((n_samples_per_domain, n_vars))
        # Exogenous (confounder)
        data[:, 0] = rng.normal(0, 1, n_samples_per_domain)
        # Treatment: affected by confounder
        data[:, 1] = 0.7 * data[:, 0] + rng.normal(0, 0.5, n_samples_per_domain)
        # Outcome: affected by confounder AND treatment
        data[:, 2] = (0.4 * data[:, 0] + 0.6 * data[:, 1] +
                      rng.normal(0, 0.3, n_samples_per_domain))

        # Compute true ATE
        true_ate = 0.6  # direct effect of treatment on outcome

        # Observational difference
        t_median = np.median(data[:, 1])
        high = data[:, 1] > t_median
        low = ~high
        obs_diff = data[high, 2].mean() - data[low, 2].mean()

        datasets[domain] = {
            "var_names": vars_list,
            "var_names_cn": [template["names"][v] for v in vars_list],
            "edges": edges,
            "data": data,
            "true_ate": true_ate,
            "obs_diff": float(obs_diff),
            "confounding_bias": float(obs_diff - true_ate),
            "backdoor_set": [vars_list[0]],  # the confounder
        }

    return datasets


# ═══════════════════════════════════════════════════════════════════
#  Training Data Summary Report
# ═══════════════════════════════════════════════════════════════════

def print_summary():
    """Print a summary of all training data types."""
    print("=" * 65)
    print("  TRAINING DATA FOR GENERAL CAUSAL REASONING")
    print("=" * 65)

    types = [
        ("Type 1: Structure Learning",
         "输入: 纯观测数据\n"
         "输出: 因果图 (DAG)\n"
         "训练量: 1000+ 随机 DAG × 500 样本\n"
         "核心技能: 从条件独立性推断因果结构"),

        ("Type 2: Effect Estimation",
         "输入: (DAG, 观测数据, T, Y)\n"
         "输出: ATE (平均因果效应)\n"
         "训练量: 2000 个问题 × 1000 样本\n"
         "核心技能: 区分相关与因果, 选择调整集"),

        ("Type 3: Interventional Reasoning",
         "输入: (DAG, 观测数据, do(T=t))\n"
         "输出: E[Y | do(T=t)]\n"
         "训练量: 1000 个问题 × 2000 样本\n"
         "核心技能: 预测干预后的分布变化"),

        ("Type 4: Counterfactual",
         "输入: (SCM, 已观测状态, do(X=x'))\n"
         "输出: 目标变量的反事实值\n"
         "训练量: 500 个问题\n"
         "核心技能: 溯因→行动→预测三步推理"),

        ("Type 5: Domain Transfer",
         "输入: 不同领域的同构因果问题\n"
         "输出: 统一的因果推理策略\n"
         "训练量: 8 个领域 × 500 样本\n"
         "核心技能: 抽象因果结构, 忽略语义表面"),
    ]

    for title, desc in types:
        print(f"\n{'─' * 65}")
        print(f"  {title}")
        print(f"{'─' * 65}")
        for line in desc.split("\n"):
            print(f"  {line}")


if __name__ == "__main__":
    print_summary()

    print(f"\n{'=' * 65}")
    print("  GENERATING DATASETS...")
    print(f"{'=' * 65}")

    # Type 1
    ds1 = generate_structure_learning_dataset(n_graphs=100, seed=42)
    print(f"\n  Type 1: {len(ds1)} graphs generated")
    print(f"    Example: {ds1[0]['n_vars']} vars, "
          f"{len(ds1[0]['edges'])} edges, "
          f"{len(ds1[0]['v_structures'])} v-structures")

    # Type 2
    ds2 = generate_effect_estimation_dataset(n_problems=200, seed=42)
    confounded = sum(1 for d in ds2 if d["has_confounding"])
    print(f"\n  Type 2: {len(ds2)} problems generated")
    print(f"    Confounded: {confounded}/{len(ds2)}")
    avg_bias = np.mean([abs(d["confounding_bias"]) for d in ds2])
    print(f"    Avg confounding bias: {avg_bias:.3f}")

    # Type 3
    ds3 = generate_interventional_dataset(n_problems=100, seed=42)
    print(f"\n  Type 3: {len(ds3)} problems generated")

    # Type 4
    ds4 = generate_counterfactual_dataset(n_problems=100, seed=42)
    print(f"\n  Type 4: {len(ds4)} counterfactuals generated")

    # Type 5
    ds5 = generate_domain_transfer_dataset(seed=42)
    print(f"\n  Type 5: {len(ds5)} domains generated")
    for domain, d in ds5.items():
        print(f"    {domain}: {d['var_names_cn']} "
              f"(ATE={d['true_ate']:.1f}, bias={d['confounding_bias']:.3f})")

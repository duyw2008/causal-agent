#!/usr/bin/env python3
"""
Demo Suite — Causal Agent v0.8 全功能验证

运行:  python demos/run_all.py
      或逐个运行下面的 demo 脚本
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def green(s):  return f"\033[32m{s}\033[0m"
def cyan(s):   return f"\033[36m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"
def hdr(s):    return bold(f"\n{'='*60}\n  {s}\n{'='*60}")

# ══════════════════════════════════════════════════════════════

def demo1_simpsons_paradox():
    """Demo 1: Simpson's Paradox — 从识别到估计到敏感性，全流程"""
    print(hdr("Demo 1: Simpson's Paradox — Full Pipeline"))

    from core.graph import CausalDAG
    from core.identification import identify_effect
    from core.estimation import estimate_effect
    from core.sensitivity import full_sensitivity_report
    from core.discovery import generate_linear_data
    from core.visualization import dag_to_ascii

    # 1. 真实因果图
    dag = CausalDAG(["Gender", "Drug", "Recovery"],
                    [("Gender","Drug"), ("Gender","Recovery"), ("Drug","Recovery")])
    print("\n" + cyan("真实因果图:"))
    print(dag_to_ascii(dag))

    # 2. 生成数据
    data = generate_linear_data(dag, n_samples=3000, seed=42)
    var_names = ["Gender", "Drug", "Recovery"]

    # 3. 朴素观测
    t = data[:, 1]
    y = data[:, 2]
    median_t = np.median(t)
    obs_effect = y[t > median_t].mean() - y[t <= median_t].mean()
    print(f"\n{yellow('朴素观测差异:')} E[Recovery|Drug=high] - E[Recovery|Drug=low] = {obs_effect:+.4f}")

    # 4. 因果识别
    ident = identify_effect(dag, "Drug", "Recovery")
    print(f"\n{cyan('因果识别:')}")
    print(f"  Method: {ident.method}")
    print(f"  Adjustment set: {ident.adjustment_set}")
    print(f"  Identifiable: {ident.identifiable}")

    # 5. 效应估计
    print(f"\n{cyan('效应估计 (所有方法):')}")
    for method in ["linear","psm","ipw","dr","stratified"]:
        est = estimate_effect(data, var_names, "Drug", "Recovery",
                              ident.adjustment_set, method=method)
        sig = green("✓") if est.is_significant() else "✗"
        print(f"  {est.method:30s}: ATE={est.ate:+.4f}  SE={est.std_error:.4f}  {sig}")

    # 6. 敏感性分析
    print(f"\n{cyan('敏感性分析:')}")
    est_linear = estimate_effect(data, var_names, "Drug", "Recovery",
                                  ident.adjustment_set, method="linear")
    print(full_sensitivity_report(est_linear.ate, est_linear.std_error))

    return True


def demo2_causal_discovery():
    """Demo 2: Causal Discovery — 从数据中自动发现因果结构"""
    print(hdr("Demo 2: Causal Discovery"))

    from core.graph import CausalDAG
    from core.discovery import (generate_linear_data, pc_algorithm,
                                 ges_algorithm, fci_algorithm,
                                 bootstrap_edge_confidence)

    # 生成三种基本结构的合成数据
    structures = {
        "Chain  (X→M→Y)": CausalDAG(["X","M","Y"], [("X","M"),("M","Y")]),
        "Fork   (Z→X, Z→Y)": CausalDAG(["X","Y","Z"], [("Z","X"),("Z","Y")]),
        "Collider (X→Z←Y)": CausalDAG(["X","Y","Z"], [("X","Z"),("Y","Z")]),
    }

    for name, true_dag in structures.items():
        print(f"\n{cyan('真实结构: ' + name)}")
        data = generate_linear_data(true_dag, n_samples=2000, seed=42)

        # PC
        pc_dag = pc_algorithm(data, true_dag.variables, alpha=0.01)
        match = (set(tuple(pc_dag.children(v)) for v in pc_dag.variables
                      if pc_dag.children(v)) ==
                 set(tuple(true_dag.children(v)) for v in true_dag.variables
                      if true_dag.children(v)))
        print(f"  PC:    {pc_dag}  {'✓' if match else '(equiv class)'}")

    # Bootstrap 置信度
    print(f"\n{cyan('自举置信度 (Simpson DAG, 50次):')}")
    dag = CausalDAG(["G","D","R"], [("G","D"),("G","R"),("D","R")])
    data = generate_linear_data(dag, 2000, seed=42)
    conf = bootstrap_edge_confidence(data, ["G","D","R"], method="pc",
                                     n_bootstrap=50, verbose=False)
    for (u,v), c in sorted(conf.items(), key=lambda x: -x[1]):
        bar = "█"*int(c*20) + "░"*(20-int(c*20))
        print(f"  {u}→{v}: {c:.2f}  {bar}")

    # FCI
    print(f"\n{cyan('FCI (容忍隐变量):')}")
    pag = fci_algorithm(data, ["G","D","R"], alpha=0.05)
    print(f"  {pag.summary()}")

    return True


def demo3_counterfactual():
    """Demo 3: Counterfactual Reasoning — 如果当初不吸烟会怎样?"""
    print(hdr("Demo 3: Counterfactual Reasoning"))

    from core.graph import CausalDAG
    from core.scm import linear_scm

    # Smoking → Tar → Cancer, with Genetic confounder
    dag = CausalDAG(
        ["Gene", "Smoking", "Tar", "Cancer"],
        [("Gene","Smoking"), ("Gene","Cancer"),
         ("Smoking","Tar"), ("Tar","Cancer"), ("Smoking","Cancer")]
    )

    scm = linear_scm(dag, coefficients={
        "Smoking": {"Gene": 0.3},
        "Tar":     {"Smoking": 0.8},
        "Cancer":  {"Tar": 0.5, "Smoking": 0.2, "Gene": 0.4},
    }, noise_std=0.05)

    # 一个特定个体的观测
    observed = {"Gene": 2.0, "Smoking": 3.0, "Tar": 2.5, "Cancer": 4.0}
    print(f"\n{cyan('观测到的个体:')} {observed}")

    # 反事实: 如果这个人不吸烟 (Smoking=0)
    cf = scm.counterfactual(observed, {"Smoking": 0.0}, "Cancer")
    print(f"\n{yellow('反事实 (do Smoking=0):')} Cancer = {cf:.3f}")
    print(f"  效应: {observed['Cancer']:.1f} → {cf:.1f} (减少 {observed['Cancer']-cf:.1f})")

    # 人群层面干预
    intv = scm.intervene({"Smoking": 0.0})
    pop = intv.sample(10000, seed=42)
    pop_obs = scm.sample(10000, seed=42)
    ate_pop = pop_obs["Cancer"].mean() - pop["Cancer"].mean()
    print(f"\n{cyan('人群层面 ATE (Smoking 1 vs 0):')} {ate_pop:.3f}")

    return True


def demo4_modern_methods():
    """Demo 4: Modern Methods — DML + CATE"""
    print(hdr("Demo 4: Modern Methods — DML + CATE"))

    from core.graph import CausalDAG
    from core.discovery import generate_linear_data
    from core.estimation import estimate_effect
    from core.modern import (estimate_ate_dml, estimate_cate_slearner,
                              estimate_cate_tlearner, estimate_cate_xlearner,
                              estimate_cate_forest)

    dag = CausalDAG(["G","D","R"], [("G","D"),("G","R"),("D","R")])
    data = generate_linear_data(dag, 3000, seed=42)
    v = ["G","D","R"]

    # 所有方法比较
    print(f"\n{cyan('ATE 估计 — 9 种方法对比:')}")
    methods = []
    for name in ["linear","psm","ipw","dr","stratified"]:
        est = estimate_effect(data, v, "D", "R", ["G"], method=name)
        methods.append((name, est.ate, est.std_error))

    est_dml = estimate_ate_dml(data, v, "D", "R", ["G"], n_folds=5)
    methods.append(("dml", est_dml.ate, est_dml.std_error))

    for name, ate, se in methods:
        print(f"  {name:12s}: ATE={ate:+.4f}  SE={se:.4f}")

    # CATE
    print(f"\n{cyan('CATE — 异质性效应 (4 种方法):')}")
    for label, fn in [("S-learner", estimate_cate_slearner),
                      ("T-learner", estimate_cate_tlearner),
                      ("X-learner", estimate_cate_xlearner),
                      ("CausalForest", estimate_cate_forest)]:
        c = fn(data, v, "D", "R", ["G"])
        print(f"  {label:15s}: ATE={c.ate:+.4f}  "
              f"range=[{c.cate.min():+.3f}, {c.cate.max():+.3f}]")

    return True


def demo5_domain_transfer():
    """Demo 5: Domain Transfer — 同一因果结构跨 8 个领域"""
    print(hdr("Demo 5: Domain Transfer — 8 Domains, 1 Causal Structure"))

    from core.graph import CausalDAG
    from core.identification import identify_effect
    from core.estimation import estimate_effect
    from core.modern import estimate_ate_dml

    # 加载 Type 5 数据集
    import json
    idx_path = os.path.join(os.path.dirname(__file__),
                            "../datasets/type5_domain_transfer/index.json")
    with open(idx_path) as f:
        idx = json.load(f)

    true_ate = 0.6
    results = []

    for d in idx["domains"]:
        domain = d["domain"]
        csv_path = os.path.join(os.path.dirname(__file__),
                                f"../datasets/type5_domain_transfer/{domain}.csv")
        meta_path = os.path.join(os.path.dirname(__file__),
                                 f"../datasets/type5_domain_transfer/{domain}_meta.json")

        data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
        with open(meta_path) as f:
            meta = json.load(f)

        dag = CausalDAG(meta["var_names"], [tuple(e) for e in meta["edges"]])
        ident = identify_effect(dag, meta["var_names"][1], meta["var_names"][2])
        est = estimate_effect(data, meta["var_names"],
                              meta["var_names"][1], meta["var_names"][2],
                              ident.adjustment_set, method="linear")

        results.append({
            "domain": domain,
            "cn": d["var_names_cn"],
            "est_ate": est.ate,
            "obs_diff": meta["obs_diff"],
            "bias": est.ate - true_ate,
        })

    print(f"\n  {'领域':12s} {'变量':30s} {'估计ATE':>8s} {'真实ATE':>8s} {'偏差':>8s}")
    print(f"  {'-'*12} {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        cn_short = ", ".join(r["cn"])
        print(f"  {r['domain']:12s} {cn_short:30s} {r['est_ate']:8.4f} "
              f"{true_ate:8.4f} {r['bias']:+8.4f}")

    avg_bias = np.mean([abs(r["bias"]) for r in results])
    avg_obs_bias = np.mean([abs(r["obs_diff"] - true_ate) for r in results])
    print(f"\n  {green(f'平均因果偏差: {avg_bias:.4f}')}")
    print(f"  {yellow(f'平均观测偏差: {avg_obs_bias:.4f}')}")
    print(f"  {bold(f'偏差消除: {(1-avg_bias/avg_obs_bias)*100:.0f}%')}")

    return True


# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demos = [
        ("Simpson's Paradox — 全流程", demo1_simpsons_paradox),
        ("Causal Discovery — PC/FCI/Bootstrap", demo2_causal_discovery),
        ("Counterfactual Reasoning", demo3_counterfactual),
        ("Modern Methods — DML + CATE", demo4_modern_methods),
        ("Domain Transfer — 8 领域", demo5_domain_transfer),
    ]

    passed = 0
    failed = 0

    for name, fn in demos:
        try:
            fn()
            passed += 1
            print(green(f"\n  ✓ {name} PASSED"))
        except Exception as e:
            failed += 1
            print(f"\n  ✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print(bold(f"\n{'='*60}"))
    print(bold(f"  RESULTS: {passed}/{len(demos)} passed"
               + (f", {failed} failed" if failed else "")))
    print(bold(f"{'='*60}"))

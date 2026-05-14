#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║           Causal Agent — 初学者交互教程                     ║
║                                                            ║
║  本教程将带你一步步体验:                                    ║
║    1. 理解因果图 (谁影响谁)                                 ║
║    2. 从数据中发现因果结构                                  ║
║    3. 估计因果效应 (影响有多大)                             ║
║    4. 敏感性分析 (结论有多可靠)                             ║
║    5. 反事实推理 (如果当初...会怎样)                        ║
║                                                            ║
║  运行: python demos/tutorial.py                            ║
╚══════════════════════════════════════════════════════════════╝
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# ══════════════════════════════════════════════════════════════
#  Helper: 彩色输出
B = "\033[1m"      # 粗体
C = "\033[36m"     # 青色
G = "\033[32m"     # 绿色
Y = "\033[33m"     # 黄色
R = "\033[0m"      # 重置

def title(s):   print(f"\n{B}{'='*55}\n  {s}\n{'='*55}{R}")
def step(n, s): print(f"\n{C}━━━ 步骤 {n}: {s} ━━━{R}")
def info(s):    print(f"  {s}")
def result(s):  print(f"{G}  → {s}{R}")
def warn(s):    print(f"{Y}  ⚠ {s}{R}")
def pause():    input(f"\n{Y}  按 Enter 继续...{R}")

# ══════════════════════════════════════════════════════════════
#  开始教程
# ══════════════════════════════════════════════════════════════

print(f"""
{B}╔══════════════════════════════════════════════════════════════╗
║           Causal Agent — 初学者交互教程                     ║
║                                                            ║
║  你将扮演一位数据分析师，任务是回答:                        ║
║                                                            ║
║  「一种新药 (Drug) 是否真的能提高康复率 (Recovery)？」      ║
║                                                            ║
║  注意: 性别 (Gender) 既影响开药概率，也影响康复率           ║
║        这就是著名的 Simpson's Paradox                       ║
╚══════════════════════════════════════════════════════════════╝{R}
""")

# ══════════════════════════════════════════════════════════════
#  步骤 1: 理解因果图
# ══════════════════════════════════════════════════════════════
step(1, "理解因果图 — 谁影响谁？")

info("因果图是用箭头表示因果关系的图:")
info("  Gender → Drug       (性别影响医生是否开药)")
info("  Gender → Recovery   (性别本身也影响康复)")
info("  Drug → Recovery     (药物对康复的效果——这是我们想知道的)")

from core.graph import CausalDAG
from core.visualization import dag_to_ascii

dag = CausalDAG(
    ["Gender", "Drug", "Recovery"],
    [("Gender", "Drug"), ("Gender", "Recovery"), ("Drug", "Recovery")]
)

info("\n用 ASCII 艺术画展示因果图:")
print(f"{C}{dag_to_ascii(dag)}{R}")

info(f"\n关键概念 — 混杂因子 (Confounder):")
info(f"  Gender 同时影响 Drug 和 Recovery，是混杂因子。")
info(f"  如果不控制 Gender，Drug 对 Recovery 的效应会被污染。")

# ══════════════════════════════════════════════════════════════
#  步骤 2: 生成数据
# ══════════════════════════════════════════════════════════════
step(2, "生成模拟数据 — 我们已知「真相」")

info("在真实世界中我们不知道真相。但这里我们先模拟一个「上帝视角」的数据。")
info("真相: Drug 对 Recovery 的真实因果效应是正面的。")

from core.discovery import generate_linear_data

data = generate_linear_data(dag, n_samples=1000, seed=42)
var_names = ["Gender", "Drug", "Recovery"]

info(f"\n数据形状: {data.shape[0]} 个患者, {data.shape[1]} 个变量")
info(f"前 5 行数据:")
for i in range(5):
    vals = ", ".join(f"{data[i,j]:+.3f}" for j in range(3))
    info(f"  [{vals}]")

# 朴素观测
t_col = data[:, 1]; y_col = data[:, 2]
median_t = np.median(t_col)
high = y_col[t_col > median_t].mean()
low = y_col[t_col <= median_t].mean()
obs_diff = high - low

info(f"\n如果直接比较 (不控制 Gender):")
info(f"  用药多的组平均康复率:  {high:+.4f}")
info(f"  用药少的组平均康复率:  {low:+.4f}")
info(f"  朴素差异:              {obs_diff:+.4f}")
warn(f"但这个差异包含混杂效应！不能直接解读为因果效应。")

# ══════════════════════════════════════════════════════════════
#  步骤 3: 因果发现
# ══════════════════════════════════════════════════════════════
step(3, "因果发现 — 让数据告诉我们因果结构")

info("假设我们不知道因果图。智能体使用 PC 算法从数据中自动发现。")

from core.discovery import pc_algorithm, bootstrap_edge_confidence

learned_dag = pc_algorithm(data, var_names, alpha=0.01)
info(f"\nPC 算法发现的因果结构: {learned_dag}")

# 比较
pc_edges = set((v,c) for v in learned_dag.variables for c in learned_dag.children(v))
true_edges = {("Gender","Drug"),("Gender","Recovery"),("Drug","Recovery")}
if pc_edges == true_edges:
    result("完美！PC 算法完全恢复了真实的因果图。")
else:
    info(f"  发现的边: {pc_edges}")
    info(f"  真实的边: {true_edges}")

info("\n自举置信度 (bootstrap): 每条边有多可靠？")
conf = bootstrap_edge_confidence(data, var_names, n_bootstrap=30)
for (u,v), c in sorted(conf.items(), key=lambda x: -x[1]):
    bar = "█" * int(c*20) + "░" * (20-int(c*20))
    info(f"  {u}→{v}: {c:.2f}  {bar}")

# ══════════════════════════════════════════════════════════════
#  步骤 4: 因果效应识别
# ══════════════════════════════════════════════════════════════
step(4, "因果效应识别 — 能不能从观测数据中估计？")

info("智能体使用 do-calculus 自动判断:")
info("  1) 这个效应能不能从观测数据中估计？")
info("  2) 如果可以，需要控制哪些变量？")

from core.identification import identify_effect

ident = identify_effect(dag, "Drug", "Recovery")
info(f"\n识别结果:")
info(f"  可识别: {ident.identifiable}")
info(f"  方法:   {ident.method}")
info(f"  调整集: {ident.adjustment_set}")
result(f"需要控制 Gender 来消除混杂效应。")

# ══════════════════════════════════════════════════════════════
#  步骤 5: 效应估计
# ══════════════════════════════════════════════════════════════
step(5, "效应估计 — 影响到底有多大？")

info("现在用不同的方法来估计 Drug 对 Recovery 的因果效应。")
info("每种方法都用 Gender 作为调整变量。")

from core.estimation import estimate_effect

info(f"\n{'方法':25s} {'ATE':>8s}  {'SE':>8s}  {'显著?':>6s}")
info(f"{'-'*25} {'-'*8} {'-'*8} {'-'*6}")

for method in ["linear", "psm", "ipw", "dr", "stratified"]:
    est = estimate_effect(data, var_names, "Drug", "Recovery",
                          ident.adjustment_set, method=method)
    sig = "✓" if est.is_significant() else "✗"
    marker = " ←" if method == "linear" else ""
    info(f"{est.method:25s} {est.ate:8.4f}  {est.std_error:8.4f}  {sig:>6s}{marker}")

est_linear = estimate_effect(data, var_names, "Drug", "Recovery",
                              ident.adjustment_set, method="linear")
result(f"\n结论: Drug 对 Recovery 有显著的因果效应。")
result(f"ATE = {est_linear.ate:.4f} ± {est_linear.std_error:.4f}")
result(f"95% 置信区间: [{est_linear.ci_lower:.4f}, {est_linear.ci_upper:.4f}]")

# ══════════════════════════════════════════════════════════════
#  步骤 6: 敏感性分析
# ══════════════════════════════════════════════════════════════
step(6, "敏感性分析 — 这个结论有多可靠？")

info("即使控制了 Gender，如果有未观测的混杂因子怎么办？")
info("敏感性分析回答: 要多强的未观测因子才能推翻我们的结论？")

from core.sensitivity import e_value, rosenbaum_bounds

ev = e_value(est_linear.ate, est_linear.std_error)
rb = rosenbaum_bounds(est_linear.ate, est_linear.std_error)

info(f"\nE-value = {ev.e_value:.2f}")
info(f"  解读: 未观测混淆因子需要同时与 Drug 和 Recovery")
info(f"        有至少 {ev.e_value:.1f} 倍的风险比关联，才能解释掉观测效应。")
info(f"  E-value > 3 → 结论稳健 ✓")

info(f"\nRosenbaum 界限: Γ 阈值 = {rb.gamma_threshold:.2f}")
info(f"  解读: 即使未观测因子使用药概率增加 {rb.gamma_threshold:.1f} 倍，")
info(f"        结论仍然统计显著。")

# ══════════════════════════════════════════════════════════════
#  步骤 7: 反事实推理
# ══════════════════════════════════════════════════════════════
step(7, "反事实推理 — 如果当初做了不同选择会怎样？")

info("反事实回答「如果...会怎样」的问题。")
info("以抽烟→焦油→癌症为例:")

from core.scm import linear_scm

dag_smoke = CausalDAG(
    ["Gene", "Smoking", "Tar", "Cancer"],
    [("Gene","Smoking"), ("Gene","Cancer"),
     ("Smoking","Tar"), ("Tar","Cancer"), ("Smoking","Cancer")]
)
scm = linear_scm(dag_smoke, coefficients={
    "Smoking": {"Gene": 0.3},
    "Tar":     {"Smoking": 0.8},
    "Cancer":  {"Tar": 0.5, "Smoking": 0.2, "Gene": 0.4},
}, noise_std=0.05)

# 一个特定个体
observed = {"Gene": 2.0, "Smoking": 3.0, "Tar": 2.5, "Cancer": 4.0}
info(f"\n观测到一个吸烟者: Gene=2.0, Smoking=3.0, Tar=2.5, Cancer=4.0")

# 反事实: 如果他当初没吸烟
cf = scm.counterfactual(observed, {"Smoking": 0.0}, "Cancer")
info(f"\n反事实 (do Smoking=0):")
result(f"  如果这个人不吸烟，预期 Cancer = {cf:.1f}")
result(f"  吸烟的因果效应 = {observed['Cancer']:.1f} - {cf:.1f} = {observed['Cancer']-cf:.1f}")
result(f"  吸烟使 Cancer 风险增加了 {observed['Cancer']-cf:.1f} 个单位")

# ══════════════════════════════════════════════════════════════
#  总结
# ══════════════════════════════════════════════════════════════
title("教程总结 — 你学到了什么")

summary = f"""
{B}1. 因果图 (CausalDAG){R}
   用箭头表示因果关系。Gender→Drug 表示性别影响用药。
   「混杂因子」是同时影响原因和结果的变量。

{B}2. 因果发现 (PC Algorithm){R}
   从纯数据中自动发现因果结构。
   自举置信度告诉你每条边有多可靠。

{B}3. 效应识别 (do-calculus){R}
   智能体自动判断效应是否可识别，并给出正确的调整集。
   Back-door 调整: 控制 {', '.join(ident.adjustment_set)} 来消除混杂。

{B}4. 效应估计 (9 种方法){R}
   线性回归最简单可靠: ATE = {est_linear.ate:.4f} ± {est_linear.std_error:.4f}
   所有方法一致指向显著的因果效应。

{B}5. 敏感性分析{R}
   E-value = {ev.e_value:.1f}: 结论高度稳健。
   需要非常强的未观测因子才能推翻。

{B}6. 反事实推理{R}
   回答「如果当初没吸烟」: Cancer 从 4.0 降到 {cf:.1f}。
   个体层面的因果效应 = {observed['Cancer']-cf:.1f}。

{B}下一步{R}
   • 在 Agent 中交互:  python agent.py
   • 加载你自己的数据:  agent.load_data("your_data.csv")
   • 提问:             effect Treatment Outcome
   • 物理约束:          physics_causal_pipeline(data, vars, T, Y)
"""
print(summary)

# ══════════════════════════════════════════════════════════════
print(f"{G}{'='*55}")
print(f"  教程完成！运行 python agent.py 开始你自己的因果分析")
print(f"{'='*55}{R}")

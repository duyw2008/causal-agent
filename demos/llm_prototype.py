#!/usr/bin/env python3
"""
Phase 4 Prototype: LLM Integration for Causal Agent

Demonstrates WHAT changes when an LLM is connected — the full
pipeline from natural language query to causal answer.

Without LLM (v0.8):  User must specify DAG, variables, treatment, outcome
With LLM (v0.9):     User asks in plain English, LLM handles the rest

This prototype uses a SIMULATED LLM (template-based) to show the
exact interface. Replace simulate_llm() with an actual API call
(OpenAI, Anthropic, Ollama, etc.) for production use.
"""

import sys, os, json, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ══════════════════════════════════════════════════════════════
#  LLM Interface (replace with real API call)
# ══════════════════════════════════════════════════════════════

def simulate_llm(prompt: str) -> str:
    """
    Simulated LLM — returns structured JSON responses based on
    template matching. Replace with:
      openai.ChatCompletion.create(model="gpt-4", messages=[...])
    or
      anthropic.Anthropic().messages.create(model="claude-3", ...)
    """
    prompt_lower = prompt.lower()

    # ── Task 1: Extract causal graph from question ──
    if "extract causal graph" in prompt_lower:
        # In reality, the LLM would parse: "Does education increase income,
        # controlling for family background?"
        if "education" in prompt_lower and "income" in prompt_lower:
            return json.dumps({
                "variables": ["FamilySES", "Education", "Income", "Ability"],
                "edges": [
                    ["FamilySES", "Education"],
                    ["FamilySES", "Income"],
                    ["Ability", "Education"],
                    ["Ability", "Income"],
                    ["Education", "Income"]
                ],
                "treatment": "Education",
                "outcome": "Income",
                "confounders": ["FamilySES", "Ability"],
                "justification": "FamilySES affects both education access and "
                    "earning potential. Ability (cognitive skills) affects "
                    "educational attainment and income independently. "
                    "Education→Income is the causal path of interest."
            })
        elif "smoking" in prompt_lower and "cancer" in prompt_lower:
            return json.dumps({
                "variables": ["Genetics", "Smoking", "Tar", "LungCancer"],
                "edges": [
                    ["Genetics", "Smoking"],
                    ["Genetics", "LungCancer"],
                    ["Smoking", "Tar"],
                    ["Tar", "LungCancer"],
                    ["Smoking", "LungCancer"]
                ],
                "treatment": "Smoking",
                "outcome": "LungCancer",
                "confounders": ["Genetics"],
                "justification": "Genetic factors influence both smoking "
                    "behavior and cancer susceptibility (confounding). "
                    "Smoking causes tar deposits which cause cancer "
                    "(front-door path via Tar)."
            })
        return json.dumps({"error": "Cannot parse causal graph"})

    # ── Task 2: Explain result in natural language ──
    if "explain the result" in prompt_lower:
        # Extract numbers from prompt
        ate_match = re.search(r"ATE\s*=\s*([-\d.]+)", prompt)
        se_match = re.search(r"SE\s*=\s*([\d.]+)", prompt)
        e_match = re.search(r"E-value\s*=\s*([\d.]+)", prompt)

        ate = float(ate_match.group(1)) if ate_match else 0
        se = float(se_match.group(1)) if se_match else 0
        ev = float(e_match.group(1)) if e_match else 0

        ci_low = ate - 1.96 * se
        ci_high = ate + 1.96 * se

        direction = "增加" if ate > 0 else "降低"
        robustness = "非常稳健" if ev > 5 else ("中等稳健" if ev > 2 else "较为脆弱")

        return f"""## 因果分析结果解读

**核心发现**: 处理变量对结果变量有{('正向' if ate > 0 else '负向')}因果效应。

**效应大小**: 平均处理效应 (ATE) = {ate:.2f}，标准误 = {se:.3f}。
这意味着处理每增加一个单位，结果预期{direction} {abs(ate):.2f} 个单位。
95% 置信区间为 [{ci_low:.2f}, {ci_high:.2f}]——我们有 95% 的把握真实效应落在这个范围内。

**统计显著性**: 该效应在 α=0.05 水平上{'显著' if abs(ate)/max(se,1e-12)>1.96 else '不显著'}。

**敏感性分析**: E-value = {ev:.1f}。这意味着要推翻我们的结论，
一个未观测的混淆因子需要同时与处理和结果都有至少 {ev:.1f} 倍的风险比关联。
这个结果是**{robustness}**的。

**实际含义**: {"如果这个因果效应是真实的，那么" + ('增加' if ate > 0 else '减少') + '处理的干预措施预计会带来显著的效果。' if abs(ate)/max(se,1e-12)>1.96 else '目前的证据不足以支持因果结论，可能需要更多数据或实验验证。'}
"""

    # ── Task 3: Suggest missing confounders ──
    if "suggest confounders" in prompt_lower:
        return """基于当前因果图的领域知识分析，我建议考虑以下可能遗漏的混杂因子：

1. **社会经济地位 (SES)**: 同时影响教育机会和收入水平，是教育→收入关系中最重要的混杂因子之一。

2. **认知能力 (Cognitive Ability)**: 高认知能力的人更可能接受更多教育，也更容易获得高收入。如果遗漏这个变量，教育对收入的效应会被高估。

3. **家庭背景**: 父母的收入和教育水平既影响子女的教育机会，也通过社会网络影响子女的就业和收入。

4. **地理位置**: 城市 vs 农村既影响教育资源的可及性，也影响就业市场的收入水平。

5. **健康状态**: 童年健康影响教育成就，成年健康影响工作能力和收入。

建议至少收集 SES 和认知能力的代理变量，加入调整集中。"""

    # ── Task 4: Counterfactual narrative ──
    if "counterfactual story" in prompt_lower:
        return """在观测到的现实中，李明（化名）家庭社会经济地位较低（SES=低），
只完成了高中学业（教育=12年），目前的年收入为 5.2 万元。

然而，在反事实世界里——假设李明在同样的家庭背景下，获得了大学教育
（教育=16年）——我们的因果模型预测他的预期年收入将达到 8.0 万元。

也就是说，大学教育为李明带来了**约 2.8 万元的年收入溢价**。
这个溢价已经排除了家庭背景的影响——我们在比较的是"同样家庭出身，
但教育水平不同"的两个人之间的收入差距。"""

    return json.dumps({"error": "Unknown task"})


# ══════════════════════════════════════════════════════════════
#  LLM-Powered Causal Agent
# ══════════════════════════════════════════════════════════════

class LLMCausalAgent:
    """Causal Agent augmented with LLM capabilities."""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self.llm = simulate_llm if use_llm else None
        self.dag = None
        self.scm = None
        self.data = None
        self.last_estimate = None

    def ask(self, question: str) -> str:
        """Process a natural language causal question end-to-end."""
        if not self.use_llm:
            return self._ask_without_llm(question)

        output = []
        output.append(f"📝 问题: {question}")
        output.append("")

        # Step 1: LLM extracts causal graph
        output.append("🔍 正在分析因果结构...")
        graph_json = self.llm(
            f"Extract causal graph from this question: {question}"
        )
        try:
            graph = json.loads(graph_json)
            if "error" in graph:
                return f"❌ {graph['error']}"

            from core.graph import CausalDAG
            self.dag = CausalDAG(graph["variables"],
                                 [tuple(e) for e in graph["edges"]])
            output.append(f"  变量: {', '.join(graph['variables'])}")
            output.append(f"  因果边: {', '.join(f'{u}→{v}' for u,v in graph['edges'])}")
            output.append(f"  处理: {graph['treatment']}, 结果: {graph['outcome']}")
            output.append(f"  混杂因子: {graph.get('confounders', [])}")
            output.append("")
        except Exception as e:
            return f"❌ 因果图解析失败: {e}"

        # Step 2: Causal identification (algorithmic — no LLM needed)
        output.append("🧮 正在识别因果效应...")
        from core.identification import identify_effect
        ident = identify_effect(self.dag, graph["treatment"], graph["outcome"])
        output.append(f"  方法: {ident.method}")
        output.append(f"  调整集: {ident.adjustment_set}")
        output.append("")

        # Step 3: Estimation (if data available, otherwise generate)
        if self.data is None:
            output.append("⚠️  无真实数据，使用合成数据演示...")
            from core.discovery import generate_linear_data
            self.data = generate_linear_data(self.dag, n_samples=2000, seed=42)

        from core.estimation import estimate_effect
        from core.sensitivity import e_value
        est = estimate_effect(self.data, graph["variables"],
                              graph["treatment"], graph["outcome"],
                              ident.adjustment_set, method="linear")
        self.last_estimate = est
        ev = e_value(est.ate, est.std_error)

        output.append(f"📊 因果效应估计:")
        output.append(f"  ATE = {est.ate:.4f}  (SE = {est.std_error:.4f})")
        output.append(f"  95% CI = [{est.ci_lower:.4f}, {est.ci_upper:.4f}]")
        output.append(f"  E-value = {ev.e_value:.2f}")
        output.append("")

        # Step 4: LLM explains the result
        output.append("💡 自然语言解读:")
        explanation = self.llm(
            f"Explain the result: ATE={est.ate:.4f}, SE={est.std_error:.4f}, "
            f"E-value={ev.e_value:.2f}, treatment={graph['treatment']}, "
            f"outcome={graph['outcome']}"
        )
        output.append(explanation)
        output.append("")

        # Step 5: LLM suggests confounders
        output.append("🔎 遗漏变量建议:")
        suggestions = self.llm(
            f"Suggest confounders for the causal graph: variables={graph['variables']}, "
            f"edges={graph['edges']}"
        )
        output.append(suggestions)

        return "\n".join(output)

    def _ask_without_llm(self, question: str) -> str:
        """Fallback: attempt rule-based parsing (v0.8 behavior)."""
        output = [f"📝 问题: {question}", ""]
        output.append("⚠️  LLM 未启用，使用规则解析（能力有限）...")
        output.append("   建议: 手动指定 DAG 使用 'load <description>' 命令")
        output.append("   或启用 LLM: agent = LLMCausalAgent(use_llm=True)")
        return "\n".join(output)

    def counterfactual_narrative(self, observed: dict, intervention: dict,
                                  target: str) -> str:
        """Generate a human-readable counterfactual story."""
        if self.scm is None or not self.use_llm:
            return "需要 SCM 和 LLM 来生成反事实叙事。"

        cf_val = self.scm.counterfactual(observed, intervention, target)
        story = self.llm(
            "Generate a counterfactual story. "
            f"Observed: {observed}. "
            f"Intervention: do({intervention}). "
            f"Counterfactual {target} = {cf_val:.2f}."
        )
        return story


# ══════════════════════════════════════════════════════════════
#  Demo
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  PHASE 4 PROTOTYPE: LLM-Powered Causal Agent")
    print("  (using simulated LLM — replace with real API)")
    print("=" * 65)

    agent = LLMCausalAgent(use_llm=True)

    # Demo: Education → Income
    print("\n" + "─" * 65)
    question = "Does education increase income, controlling for family background and ability?"
    response = agent.ask(question)
    print(response)

    # Show what happens WITHOUT LLM
    print("\n" + "─" * 65)
    print("  WITHOUT LLM (v0.8 behavior):")
    print("─" * 65)
    agent_no_llm = LLMCausalAgent(use_llm=False)
    print(agent_no_llm.ask(question))

    print("\n" + "=" * 65)
    print("  LLM integration adds 4 key capabilities:")
    print("    1. Free-text → causal graph (no manual DAG needed)")
    print("    2. Technical results → natural language explanation")
    print("    3. Domain knowledge → confounder suggestions")
    print("    4. Counterfactual → human-readable narratives")
    print("=" * 65)

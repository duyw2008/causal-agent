"""
DeepSeek LLM Client — 自然语言 → 因果模型

通过 DeepSeek API 实现三项能力：
  1. 自由文本 → 因果图 (extract_causal_graph)
  2. 分析结果 → 自然语言解读 (explain_result)
  3. 反事实场景 → 叙事生成 (generate_counterfactual_narrative)

API 兼容 OpenAI SDK 格式。
配置: export DEEPSEEK_API_KEY="sk-..."
"""

from __future__ import annotations
import json, os, re, urllib.request, urllib.error
from typing import Any, Dict, Optional


# ═══════════════════════════════════════════════════════════════════
#  API Client
# ═══════════════════════════════════════════════════════════════════

class DeepSeekClient:
    """Minimal DeepSeek API client using urllib (no external deps)."""

    BASE_URL = "https://api.deepseek.com/v1"
    CONFIG_PATH = os.path.expanduser("~/.hermes/causal_config.json")

    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        self.model = model
        # Fallback to config file
        if not self.api_key:
            self.api_key = self._load_key_from_config()

    def _load_key_from_config(self) -> str:
        """Load API key from config file."""
        try:
            if os.path.exists(self.CONFIG_PATH):
                with open(self.CONFIG_PATH) as f:
                    cfg = json.load(f)
                return cfg.get("DEEPSEEK_API_KEY", "")
        except Exception:
            pass
        return ""

    def chat(self, messages: list, temperature: float = 0.1,
             max_tokens: int = 2048) -> str:
        """Send a chat completion request. Returns the response text."""
        if not self.api_key:
            return json.dumps({"error": "DEEPSEEK_API_KEY not set. "
                               "Run: export DEEPSEEK_API_KEY='sk-...'"})

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.BASE_URL}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err = json.loads(e.read().decode("utf-8"))
            return json.dumps({"error": f"API error {e.code}: {err}"})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ═══════════════════════════════════════════════════════════════════
#  Prompt Templates
# ═══════════════════════════════════════════════════════════════════

CAUSAL_GRAPH_PROMPT = """You are a causal inference expert. Given a user's question in natural language, extract the causal graph structure.

Return ONLY valid JSON with these fields:
{{"variables": ["list", "of", "variable", "names"], "edges": [["cause", "effect"]], "treatment": "string — the intervention variable", "outcome": "string — the outcome variable", "confounders": ["list", "of", "confounding", "variables"], "justification": "brief explanation of the causal structure"}}

Rules:
- Variable names should be short CamelCase identifiers (e.g. Education, Income, FamilySES)
- Edges represent causal direction: ["cause", "effect"]
- Confounders are variables that affect BOTH treatment and outcome
- If the user asks about an effect, infer treatment and outcome from context
- Consider realistic confounding variables from domain knowledge

User question: {question}

JSON:"""


EXPLAIN_RESULT_PROMPT = """You are a bilingual (Chinese/English) causal inference communicator. Given technical analysis results, write a clear, accessible explanation in Chinese.

Analysis context:
- Treatment: {treatment}
- Outcome: {outcome}
- Variables: {variables}
- Causal DAG: {dag_summary}
- Identification method: {method}
- Adjustment set: {adjustment_set}

Results:
- ATE = {ate:.4f} (Average Treatment Effect)
- Standard Error = {se:.4f}
- 95% CI = [{ci_low:.4f}, {ci_high:.4f}]
- E-value = {evalue:.2f}
- Interpretation: {interpretation}

Write a natural language explanation covering:
1. What the causal effect means in plain language
2. How confident we are (CI interpretation)
3. What confounders were controlled for
4. How robust the finding is (E-value)
5. One practical implication or recommendation

Respond in Chinese, 3-5 paragraphs, conversational tone."""


COUNTERFACTUAL_PROMPT = """You are a causal reasoning narrator. Given a counterfactual analysis, write a compelling narrative.

Observed reality:
{observed}

Intervention (what if):
{intervention}

Counterfactual outcome:
{counterfactual}

Write a short narrative (3-5 sentences) in Chinese that contrasts "what actually happened" with "what would have happened" under the counterfactual scenario. Make it concrete and relatable, not technical."""


# ═══════════════════════════════════════════════════════════════════
#  High-Level API
# ═══════════════════════════════════════════════════════════════════

class LLMCausalInterface:
    """Bridge between LLM and causal agent."""

    def __init__(self, client: Optional[DeepSeekClient] = None):
        self.client = client or DeepSeekClient()

    def extract_causal_graph(self, question: str) -> Dict[str, Any]:
        """Parse a natural language question into a causal graph."""
        prompt = CAUSAL_GRAPH_PROMPT.format(question=question)
        response = self.client.chat([
            {"role": "user", "content": prompt}
        ], temperature=0.0, max_tokens=1024)

        try:
            # Extract JSON from response (may contain markdown fences)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            return {"error": "Could not parse JSON from response",
                    "raw_response": response[:300]}
        except json.JSONDecodeError:
            return {"error": "JSON decode failed",
                    "raw_response": response[:300]}

    def explain_result(
        self,
        treatment: str,
        outcome: str,
        variables: str,
        dag_summary: str,
        method: str,
        adjustment_set: str,
        ate: float,
        se: float,
        evalue: float,
        interpretation: str = "",
    ) -> str:
        """Generate a natural language explanation of causal analysis results."""
        ci_low = ate - 1.96 * se
        ci_high = ate + 1.96 * se

        prompt = EXPLAIN_RESULT_PROMPT.format(
            treatment=treatment, outcome=outcome, variables=variables,
            dag_summary=dag_summary, method=method,
            adjustment_set=adjustment_set, ate=ate, se=se,
            ci_low=ci_low, ci_high=ci_high, evalue=evalue,
            interpretation=interpretation,
        )
        return self.client.chat([
            {"role": "user", "content": prompt}
        ], temperature=0.3, max_tokens=1500)

    def generate_counterfactual_narrative(
        self,
        observed: str,
        intervention: str,
        counterfactual: str,
    ) -> str:
        """Generate a narrative for a counterfactual scenario."""
        prompt = COUNTERFACTUAL_PROMPT.format(
            observed=observed,
            intervention=intervention,
            counterfactual=counterfactual,
        )
        return self.client.chat([
            {"role": "user", "content": prompt}
        ], temperature=0.5, max_tokens=800)


# ═══════════════════════════════════════════════════════════════════
#  Test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("DEEPSEEK_API_KEY not set.  Set it to test:")
        print("  export DEEPSEEK_API_KEY='sk-...'")
        print("  python core/llm_client.py")
        sys.exit(0)

    client = DeepSeekClient(api_key=api_key)
    llm = LLMCausalInterface(client)

    print("=" * 60)
    print("  DeepSeek LLM Client — Smoke Test")
    print("=" * 60)

    # Test 1: Extract causal graph
    question = ("Does education increase income, "
                "controlling for family background and cognitive ability?")
    print(f"\n  Query: {question}")
    result = llm.extract_causal_graph(question)
    if "error" in result:
        print(f"  ✗ Error: {result['error']}")
        if "raw_response" in result:
            print(f"  Raw: {result['raw_response']}")
    else:
        print(f"  ✓ Variables: {result.get('variables')}")
        print(f"  ✓ Edges: {result.get('edges')}")
        print(f"  ✓ Treatment: {result.get('treatment')}")
        print(f"  ✓ Outcome: {result.get('outcome')}")
        print(f"  ✓ Confounders: {result.get('confounders')}")

    # Test 2: Explain result
    explanation = llm.explain_result(
        treatment="Education", outcome="Income",
        variables="FamilySES, Education, Income, Ability",
        dag_summary="FamilySES→Education, FamilySES→Income, Ability→Education, Ability→Income, Education→Income",
        method="Back-door adjustment",
        adjustment_set="FamilySES, Ability",
        ate=2.3, se=0.4, evalue=3.5,
    )
    print(f"\n  ── Explanation ──")
    print(f"  {explanation[:300]}...")
    print(f"\n  ✓ LLM client operational")

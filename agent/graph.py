"""
agent/graph.py — Pixellot LangGraph State Machine
--------------------------------------------------

Graph topology:

    START
      │
    analyzer  ──────────────────────────────────────────────┐
      │  (should_handoff=False)              (should_handoff=True)
      │                                                      │
    knowledge_retriever                              handoff_guard
      │                                                      │
    generator                                               END
      │
     END

Nodes:
  analyzer          – structured LLM call: detects language + intent
  knowledge_retriever – keyword-intent retrieval from KB (no LLM)
  generator         – final LLM response grounded in KB context
  handoff_guard     – deterministic bilingual redirect (no LLM)
"""
from __future__ import annotations

import re
from typing import List, Literal, Optional, Tuple, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from agent.kb import Intent, _detect_intent, retrieve

# ── Constants ─────────────────────────────────────────────────────────────────

_HEBREW_RE = re.compile(r"[\u0590-\u05FF\uFB1D-\uFB4F]")

_HANDOFF_TEMPLATES = {
    "en": (
        "Great question! For exact pricing and purchasing options, our sales team can give you "
        "the most accurate and personalised information.\n\n"
        "I'd recommend reaching out directly:\n"
        "- **Email:** sales@pixellot.tv\n"
        "- **Website:** pixellot.tv/contact\n\n"
        "You can request a live demo and a custom quote there. "
        "Is there anything about Pixellot's technology or products I can help with in the meantime?"
    ),
    "he": (
        "שאלה מצוינת! לפרטי מחירים ואפשרויות רכישה, צוות המכירות של Pixellot יוכל לתת לך את המידע המדויק ביותר.\n\n"
        "אני ממליץ לפנות ישירות:\n"
        "- **אימייל:** sales@pixellot.tv\n"
        "- **אתר:** pixellot.tv/contact\n\n"
        "שם תוכל לבקש הדגמה חיה ואומדן מחיר מותאם אישית. "
        "האם יש עוד משהו לגבי הטכנולוגיה או המוצרים של Pixellot שאוכל לעזור לך בינתיים?"
    ),
}

_SYSTEM_PROMPT_TEMPLATE = """You are the **Pixellot AI Knowledge Assistant** — a helpful, professional, and enthusiastic agent representing Pixellot, the world's leading AI-automated sports video platform.

Your personality:
- Knowledgeable but approachable
- Sports-tech savvy
- Concise and to the point — athletes and coaches don't have time to waste

Your responsibilities:
- Answer questions about Pixellot's products: Air NXT, Show, Prime, DoublePlay, OTT, Automatic Highlights, Pixellot You, Analytics (VidSwap/Advantage)
- Explain AI features, hardware specs, supported sports, installation, and integrations
- Help organizations (schools, clubs, leagues, broadcasters) understand which Pixellot solution fits their need
- Highlight Pixellot's competitive advantages and customer success stories

STRICT RULES:
1. ONLY use facts from the CONTEXT block below. Never invent product names, prices, or technical specs.
2. If a fact isn't in the CONTEXT, say: "I don't have that specific detail — I recommend reaching out to Pixellot directly at pixellot.tv/contact."
3. NEVER quote an exact price or subscription fee. If asked about cost, route to the sales team.
4. Respond ONLY in {language}. Never mix languages.
5. Keep responses focused: 2–4 paragraphs maximum. Use **bold** for product names and key numbers.
6. Use bullet points when listing features or options.

CONTEXT:
{context}
"""


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    user_message: str
    history: List[Tuple[str, str]]
    language: str           # "en" | "he"
    intent: str             # Intent literal
    relevant_context: str
    should_handoff: bool
    response: str


# ── Structured output schema ──────────────────────────────────────────────────

class AnalysisOutput(BaseModel):
    language: Literal["en", "he"]
    intent: Literal["TECH", "SALES", "PRICING", "SUPPORT", "GENERAL", "HANDOFF"]
    should_handoff: bool


# ── Agent graph ───────────────────────────────────────────────────────────────

class PixellotAgentGraph:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.structured_llm = self.llm.with_structured_output(AnalysisOutput)
        self.graph = self._build()

    # ── Node: analyzer ────────────────────────────────────────────────────────

    def _analyzer_node(self, state: AgentState) -> AgentState:
        """
        Structured LLM call: detects language and intent.
        Falls back to regex + keyword detection on any error.
        """
        msg = state["user_message"]
        try:
            result: AnalysisOutput = self.structured_llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "You are a fast intent and language classifier. "
                            "Given the user message, respond with:\n"
                            "- language: 'he' if the message contains Hebrew characters, else 'en'\n"
                            "- intent: one of TECH | SALES | PRICING | SUPPORT | GENERAL | HANDOFF\n"
                            "  HANDOFF  = user wants to buy, get exact pricing, speak to a human, or request a quote\n"
                            "  PRICING  = asking about cost or price range\n"
                            "  TECH     = asking about hardware specs, AI, installation, or integrations\n"
                            "  SALES    = asking for a demo, trial, or how to get started\n"
                            "  SUPPORT  = asking about help, issues, or FAQ\n"
                            "  GENERAL  = everything else (product overview, sports, markets)\n"
                            "- should_handoff: true ONLY for HANDOFF intent"
                        )
                    ),
                    HumanMessage(content=msg),
                ]
            )
            return {
                **state,
                "language": result.language,
                "intent": result.intent,
                "should_handoff": result.should_handoff,
            }
        except Exception:
            # Graceful fallback — ensures the graph never crashes on LLM errors
            lang = "he" if _HEBREW_RE.search(msg) else "en"
            intent = _detect_intent(msg)
            return {
                **state,
                "language": lang,
                "intent": intent,
                "should_handoff": intent == "HANDOFF",
            }

    # ── Node: knowledge_retriever ─────────────────────────────────────────────

    def _retriever_node(self, state: AgentState) -> AgentState:
        """No LLM call — pure keyword-intent retrieval from JSON KB."""
        context = retrieve(state["user_message"], state["intent"])  # type: ignore[arg-type]
        return {**state, "relevant_context": context}

    # ── Node: generator ───────────────────────────────────────────────────────

    def _generator_node(self, state: AgentState) -> AgentState:
        """Grounded LLM response using retrieved context + conversation history."""
        lang_label = "Hebrew" if state["language"] == "he" else "English"
        system_content = _SYSTEM_PROMPT_TEMPLATE.format(
            language=lang_label,
            context=state["relevant_context"],
        )

        msgs: list = [SystemMessage(content=system_content)]

        # Include last 6 turns of history for context continuity
        for role, content in state["history"][-6:]:
            if role == "user":
                msgs.append(HumanMessage(content=content))
            else:
                msgs.append(AIMessage(content=content))

        msgs.append(HumanMessage(content=state["user_message"]))

        response = self.llm.invoke(msgs)
        return {**state, "response": response.content}

    # ── Node: handoff_guard ───────────────────────────────────────────────────

    def _handoff_node(self, state: AgentState) -> AgentState:
        """
        Deterministic bilingual handoff redirect — NO LLM call.
        This is the safety gate: Pixellot sales handles pricing, not the AI.
        """
        lang = state.get("language", "en")
        template = _HANDOFF_TEMPLATES.get(lang, _HANDOFF_TEMPLATES["en"])
        return {**state, "response": template}

    # ── Router ────────────────────────────────────────────────────────────────

    @staticmethod
    def _route_after_analyzer(state: AgentState) -> str:
        return "handoff_guard" if state["should_handoff"] else "knowledge_retriever"

    # ── Graph builder ─────────────────────────────────────────────────────────

    def _build(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("analyzer", self._analyzer_node)
        workflow.add_node("knowledge_retriever", self._retriever_node)
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("handoff_guard", self._handoff_node)

        workflow.set_entry_point("analyzer")

        workflow.add_conditional_edges(
            "analyzer",
            self._route_after_analyzer,
            {
                "handoff_guard": "handoff_guard",
                "knowledge_retriever": "knowledge_retriever",
            },
        )

        workflow.add_edge("knowledge_retriever", "generator")
        workflow.add_edge("generator", END)
        workflow.add_edge("handoff_guard", END)

        return workflow.compile()

    # ── Public invoke ─────────────────────────────────────────────────────────

    async def invoke(
        self,
        user_message: str,
        history: List[Tuple[str, str]],
    ) -> dict:
        init_state: AgentState = {
            "user_message": user_message,
            "history": history,
            "language": "en",
            "intent": "GENERAL",
            "relevant_context": "",
            "should_handoff": False,
            "response": "",
        }
        return await self.graph.ainvoke(init_state)

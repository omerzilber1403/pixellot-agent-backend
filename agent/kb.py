"""
agent/kb.py — Pixellot Knowledge Base Retriever
------------------------------------------------
Keyword-intent based retrieval from pixellot_kb.json.
No embeddings — fast and deterministic.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

KB_PATH = Path(__file__).parent.parent / "pixellot_kb.json"
MAX_CONTEXT_CHARS = 9_000

Intent = Literal["TECH", "SALES", "PRICING", "SUPPORT", "GENERAL", "HANDOFF"]

# ── Keyword maps ──────────────────────────────────────────────────────────────

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "HANDOFF": [
        "buy now", "i want to buy", "i'd like to buy", "purchase", "i need to purchase",
        "how much does it cost", "exact price", "pricing plan", "get a quote",
        "talk to sales", "speak to sales", "speak with someone", "contact sales",
        "talk to a human", "human agent", "representative", "sales rep",
        "אני רוצה לקנות", "אני מעוניין לרכוש", "כמה זה עולה", "פרטי מחיר",
        "נציג מכירות", "נציג אנושי", "רוצה לדבר עם",
    ],
    "PRICING": [
        "price", "cost", "pricing", "subscription", "plan", "tier", "fee",
        "affordable", "budget", "how much", "monthly", "annual",
        "מחיר", "עלות", "מנוי", "תוכנית", "כמה עולה",
    ],
    "TECH": [
        "camera", "resolution", "bandwidth", "api", "sdk", "rtmp", "rtsp", "sdi", "ndi",
        "ai", "algorithm", "hardware", "spec", "battery", "4g", "5g", "lte", "wifi",
        "fps", "1080p", "hd", "stream", "latency", "install", "setup", "integration",
        "mount", "panoramic", "tracking", "auto", "wifi", "weight", "portable",
        "מצלמה", "טכנולוגיה", "התקנה", "רזולוציה",
    ],
    "SALES": [
        "demo", "trial", "get started", "free trial", "sign up", "book",
        "contact", "learn more", "request", "interested",
        "הדגמה", "ניסיון", "להתחיל", "לפנות",
    ],
    "SUPPORT": [
        "support", "help", "issue", "problem", "broken", "error",
        "troubleshoot", "faq", "not working", "maintenance",
        "תמיכה", "עזרה", "בעיה", "תקלה",
    ],
}

_PRODUCT_KEYWORDS: dict[str, list[str]] = {
    "air": ["air nxt", "air", "portable", "lightweight", "tripod", "under 2kg"],
    "show": ["pixellot show", " show ", "fixed camera", "ceiling", "rail mount"],
    "prime": ["prime", "broadcast", "professional", "50fps", "60fps", "la liga", "serie a"],
    "doubleplay": ["doubleplay", "double play", "baseball", "softball", "batter", "pitcher"],
    "ott": ["ott", "streaming platform", "white-label", "white label", "monetization", "channel"],
    "highlights": ["highlight", "clip", "reel", "social media", "tiktok", "shorts", "reels"],
    "you": ["pixellot you", "action camera", "bring your own", "my own camera"],
}

_SPORT_KEYWORDS = [
    "soccer", "football", "basketball", "ice hockey", "field hockey", "baseball",
    "softball", "american football", "rugby", "lacrosse", "volleyball", "handball",
    "futsal", "netball", "tennis", "wrestling", "beach volleyball", "roller hockey",
    "כדורגל", "כדורסל", "הוקי",
]

_MARKET_KEYWORDS: dict[str, list[str]] = {
    "high_school": ["high school", "school", "prep", "varsity", "תיכון"],
    "college": ["college", "university", "ncaa", "collegiate", "אוניברסיטה"],
    "professional": ["professional", "pro", "league", "federation", "liiga", "bundesliga",
                     "serie a", "la liga", "מקצועי", "ליגה"],
    "club_academy": ["club", "academy", "youth", "amateur", "קבוצה", "אקדמיה"],
    "broadcaster": ["broadcaster", "media", "tv", "television", "שידור"],
}


# ── KB loader ─────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_kb() -> dict:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Serializer ────────────────────────────────────────────────────────────────

def _ser(obj, depth: int = 0, max_list: int = 6) -> str:
    """Compact human-readable serializer for nested KB structures."""
    if depth > 3:
        return str(obj)[:300]
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            lines.append(f"{'  ' * depth}{k}: {_ser(v, depth + 1, max_list)}")
        return "\n".join(lines)
    if isinstance(obj, list):
        items = [_ser(i, depth, max_list) for i in obj[:max_list]]
        if len(obj) > max_list:
            items.append(f"... and {len(obj) - max_list} more")
        return ", ".join(items)
    return str(obj)


# ── Detectors ─────────────────────────────────────────────────────────────────

def _detect_intent(query: str) -> Intent:
    q = query.lower()
    for intent, kws in _INTENT_KEYWORDS.items():
        if any(kw in q for kw in kws):
            return intent  # type: ignore[return-value]
    return "GENERAL"


def _detect_products(query: str) -> list[str]:
    q = query.lower()
    return [prod for prod, kws in _PRODUCT_KEYWORDS.items() if any(kw in q for kw in kws)]


def _detect_sports(query: str) -> list[str]:
    q = query.lower()
    return [s for s in _SPORT_KEYWORDS if s in q]


def _detect_markets(query: str) -> list[str]:
    q = query.lower()
    return [mkt for mkt, kws in _MARKET_KEYWORDS.items() if any(kw in q for kw in kws)]


# ── Public retriever ──────────────────────────────────────────────────────────

def retrieve(query: str, intent: Optional[Intent] = None) -> str:
    """
    Returns a context string from the KB, capped at MAX_CONTEXT_CHARS.
    Intent is auto-detected if not supplied by the LangGraph analyzer node.
    """
    kb = _load_kb()
    if intent is None:
        intent = _detect_intent(query)

    products_hit = _detect_products(query)
    sports_hit = _detect_sports(query)
    markets_hit = _detect_markets(query)

    sections: list[str] = []

    # ── Always: company overview ──────────────────────────────────────────────
    company = kb.get("company", {})
    overview = {
        "name": company.get("name"),
        "tagline": company.get("tagline"),
        "mission": company.get("mission_statement"),
        "value_proposition": company.get("value_proposition"),
        "scale": company.get("scale_stats"),
        "awards": company.get("awards_recognition", [])[:3],
        "partners": company.get("social_proof_logos", [])[:10],
    }
    sections.append(f"=== COMPANY OVERVIEW ===\n{_ser(overview)}")

    # ── All products flat list ────────────────────────────────────────────────
    all_products: list[dict] = []
    for cat in ["fixed_camera_systems", "mobile_camera_systems", "software_and_platforms"]:
        all_products.extend(kb.get("products", {}).get(cat, []))

    if products_hit:
        # Full detail for matched products
        for prod in all_products:
            name = prod.get("product_name", "").lower()
            if any(p in name for p in products_hit):
                sections.append(
                    f"=== PRODUCT: {prod.get('product_name')} ===\n{_ser(prod)}"
                )
    elif intent == "TECH":
        # All products with full detail
        for prod in all_products:
            sections.append(f"=== PRODUCT: {prod.get('product_name')} ===\n{_ser(prod)}")
    else:
        # Summary for general/sales queries
        summaries = [
            f"- **{p.get('product_name')}**: {p.get('description', '')[:180]}"
            for p in all_products
        ]
        sections.append("=== PRODUCTS OVERVIEW ===\n" + "\n".join(summaries))

    # ── Intent-specific sections ──────────────────────────────────────────────
    if intent == "PRICING":
        pricing = kb.get("pricing_and_business_model", {})
        sections.append(f"=== PRICING & BUSINESS MODEL ===\n{_ser(pricing)}")

    if intent in ("SALES", "HANDOFF"):
        contact = kb.get("sales_and_contact", {})
        sections.append(f"=== CONTACT / SALES ===\n{_ser(contact)}")

    if intent == "SUPPORT":
        faq = kb.get("faq", [])
        sections.append(f"=== FAQ ===\n{_ser(faq)}")

    # ── Sport-specific context ────────────────────────────────────────────────
    if sports_hit:
        sports_data = kb.get("supported_sports", {})
        for key, val in sports_data.items():
            if any(s in key.lower() for s in sports_hit):
                sections.append(f"=== SPORT: {key} ===\n{_ser(val)}")

    # ── Market-specific context ───────────────────────────────────────────────
    if markets_hit:
        markets = kb.get("solutions_by_market", {})
        for mkt in markets_hit:
            if mkt in markets:
                sections.append(f"=== MARKET: {mkt} ===\n{_ser(markets[mkt])}")
    elif intent == "SALES":
        markets = kb.get("solutions_by_market", {})
        if markets:
            sections.append(f"=== MARKETS SERVED ===\n{_ser(markets)}")

    # ── Objection handling (sales context only) ───────────────────────────────
    if intent in ("SALES", "PRICING"):
        objections = kb.get("objection_handling", {})
        if objections:
            sections.append(f"=== OBJECTION HANDLING ===\n{_ser(objections)}")

    context = "\n\n".join(sections)
    return context[:MAX_CONTEXT_CHARS]

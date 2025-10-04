import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


class OpenAIAlertClient:
    """
    Thin wrapper around the OpenAI Chat Completions API to turn structured
    supply-chain signals into plain-English risk alerts.

    Usage:
        client = OpenAIAlertClient()
        alerts = client.generate_alerts(context)
    """

    def __init__(self, model: Optional[str] = None) -> None:
        load_dotenv(override=True)
        self.api_key = os.getenv("OPENAI_API_KEY")
        # Default to a fast, cost-effective model; allow override via env or arg
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # Tracks whether the last generate_alerts call used OpenAI (vs fallback)
        self.last_used_openai: bool = False
        self.init_error: Optional[str] = None
        self.last_error: Optional[str] = None

        # Defer importing the SDK to allow environments without the package
        self._client = None
        if self.api_key:
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI(api_key=self.api_key)
            except Exception as exc:  # pragma: no cover - import/runtime env specific
                # If SDK import fails, leave client as None; caller can fall back
                self._client = None
                self.init_error = str(exc)
        else:
            self.init_error = "Missing OPENAI_API_KEY"

    def is_available(self) -> bool:
        return self._client is not None

    def generate_alerts(self, context: Dict[str, Any], max_alerts: int = 8) -> List[str]:
        """
        Convert structured context into concise, plain-English risk alerts.

        Returns a list of alert strings. If OpenAI is not configured, returns
        a minimal rule-based set derived from the context.
        """
        # default assumption until proven otherwise
        self.last_used_openai = False
        self.last_error = None
        if not self.is_available():
            return self._rule_based_alerts(context, max_alerts=max_alerts)

        system = (
            "You are a supply-chain risk analyst. You produce concise, plain-English "
            "alerts for SMEs about cost spikes, supplier reliability issues, and market "
            "disruptions. You never invent data; you only summarize the provided context. "
            "Each alert should be one sentence, start with an emoji tag, and be actionable. "
            "Avoid duplications; prefer the highest-impact insights."
        )

        user = {
            "instruction": (
                "Generate up to {max_alerts} alerts prioritizing high-severity items. "
                "Prefer: supplier high risk, deteriorating on-time rates, rising defect rates, "
                "country news spikes, and large FX swings affecting EUR-based purchases."
            ).format(max_alerts=max_alerts),
            "context": context,
            "output_format": (
                "Return a JSON object with an 'alerts' array of strings. Do not include other keys."
            ),
            "style": {
                "tone": "professional, concise",
                "max_chars_per_alert": 180,
                "emoji_prefix_examples": ["⚠️", "📉", "📈", "💱", "🚢", "🏭"],
            },
        }

        try:
            response = self._client.chat.completions.create(  # type: ignore[attr-defined]
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": str(user)},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content  # type: ignore[index]
            if not content:
                return self._rule_based_alerts(context, max_alerts=max_alerts)

            import json

            parsed = json.loads(content)
            alerts = parsed.get("alerts") or []
            if isinstance(alerts, list):
                self.last_used_openai = True
                return [str(a) for a in alerts][:max_alerts]
            return self._rule_based_alerts(context, max_alerts=max_alerts)
        except Exception as exc:
            self.last_error = str(exc)
            return self._rule_based_alerts(context, max_alerts=max_alerts)

    def diagnostics(self) -> Dict[str, Any]:
        """Return debug information for UI display."""
        return {
            "api_key_present": bool(self.api_key),
            "model": self.model,
            "is_available": self.is_available(),
            "init_error": self.init_error,
            "last_error": self.last_error,
            "last_used_openai": self.last_used_openai,
        }

    def _rule_based_alerts(self, context: Dict[str, Any], max_alerts: int = 8) -> List[str]:
        """Fallback summaries without LLM access."""
        alerts: List[str] = []

        # Supplier high risk
        for s in context.get("suppliers_high_risk", [])[: max(3, max_alerts // 2)]:
            supplier_name = s.get("supplier_name", "Unknown")
            country = s.get("country", "-")
            score = s.get("risk_score")
            top_factor = s.get("top_factor", "risk factors")
            severity_label = s.get("news_severity_label") or s.get("news_severity") or "-"
            try:
                score_str = f"{float(score):.1f}" if score is not None else "-"
            except Exception:
                score_str = "-"
            alerts.append(
                (
                    f"⚠️ High supplier risk: {supplier_name} ({country}) score {score_str}; "
                    f"driver: {top_factor}; news {severity_label}."
                )
            )

        # Country news spikes
        for c in context.get("country_news_hotspots", [])[:2]:
            country = c.get("country", "-")
            score = c.get("news_score", 0.0)
            label = c.get("severity_label", "-")
            try:
                score_pct = f"{float(score) * 100:.0f}%"
            except Exception:
                score_pct = "-"
            alerts.append(
                (
                    f"🚨 Elevated country risk: {country} ({label}, {score_pct}); monitor shipments and lead times."
                )
            )

        # FX swings
        for fx in context.get("forex_spikes", [])[:2]:
            symbol = f"{fx.get('currency')}/{fx.get('base_currency', 'EUR')}"
            alerts.append(
                (
                    f"💱 FX volatility: {symbol} moved {fx.get('pct_change_1d', 0):+.2f}% d/d; "
                    f"hedge exposures or review pricing."
                )
            )

        return alerts[:max_alerts]



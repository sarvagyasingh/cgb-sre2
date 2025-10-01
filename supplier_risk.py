"""Utilities for supplier risk scoring and explainability."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


NEGATIVE_KEYWORDS: Dict[str, float] = {
    "strike": 1.0,
    "protest": 0.9,
    "tariff": 0.6,
    "sanction": 1.0,
    "shortage": 0.9,
    "delay": 0.6,
    "inflation": 0.5,
    "recession": 0.7,
    "unrest": 1.0,
    "earthquake": 1.0,
    "storm": 0.7,
    "flood": 0.7,
    "shutdown": 0.8,
    "fire": 0.7,
    "outage": 0.6,
    "labor": 0.5,
    "labour": 0.5,
    "political": 0.6,
    "currency": 0.4,
}

POSITIVE_KEYWORDS: Dict[str, float] = {
    "investment": 0.4,
    "expansion": 0.3,
    "growth": 0.3,
    "upgrade": 0.2,
    "stable": 0.2,
    "resilience": 0.3,
    "recovery": 0.3,
    "infrastructure": 0.2,
}

# Default weights for the explainable scoring model.
RISK_WEIGHTS: Dict[str, float] = {
    "delivery": 0.25,
    "on_time": 0.2,
    "quality": 0.2,
    "volatility": 0.15,
    "spend": 0.1,
    "news": 0.1,
}


@dataclass
class SupplierRiskResult:
    """Container for a supplier risk score and its explainability details."""

    supplier_id: str
    supplier_name: str
    country: str
    metrics: Dict[str, float]
    components: Dict[str, float]

    @property
    def risk_score(self) -> float:
        return float(sum(self.components.values()))

    @property
    def risk_level(self) -> str:
        score = self.risk_score
        if score >= 66:
            return "High"
        if score >= 33:
            return "Medium"
        return "Low"

    @property
    def top_factors(self) -> List[Tuple[str, float]]:
        return sorted(self.components.items(), key=lambda item: item[1], reverse=True)


def load_supplier_dataset(path: str) -> pd.DataFrame:
    """Load the supplier time-series dataset with standardised schema."""

    df = pd.read_csv(path, parse_dates=["order_date"])
    df["on_time_delivery_rate"] = df["on_time_delivery_rate"].clip(0, 1)
    df["defect_pct"] = df["defect_pct"].clip(lower=0)
    return df


def _min_max_scale(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    min_val = series.min()
    max_val = series.max()
    if np.isclose(max_val, min_val):
        return pd.Series(0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def aggregate_supplier_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactional data into supplier level metrics."""

    grouped = (
        df.groupby(["supplier_id", "supplier_name", "country"], dropna=False)
        .agg(
            total_orders=("order_id", "nunique"),
            total_spend=("total_price", "sum"),
            avg_price_per_unit=("price_per_unit", "mean"),
            price_volatility=("price_per_unit", "std"),
            avg_delivery_time=("delivery_time_days", "mean"),
            on_time_rate=("on_time_delivery_rate", "mean"),
            defect_rate=("defect_pct", "mean"),
            last_order_date=("order_date", "max"),
        )
        .reset_index()
    )

    grouped["price_volatility"] = grouped["price_volatility"].fillna(0)
    grouped["spend_share"] = grouped["total_spend"] / grouped["total_spend"].sum()
    return grouped


def compute_country_news_scores(news_by_country: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Score the riskiness of news headlines by country."""

    scores: Dict[str, Dict] = {}
    for country, articles in news_by_country.items():
        if not articles:
            scores[country] = {
                "score": 0.0,
                "severity": "Low",
                "keywords": [],
                "article_count": 0,
            }
            continue

        total_score = 0.0
        keywords_found: Dict[str, float] = {}
        for article in articles:
            text_parts = [
                article.get("title", ""),
                article.get("description", ""),
                article.get("content", ""),
            ]
            text = " ".join(filter(None, text_parts)).lower()
            for keyword, weight in NEGATIVE_KEYWORDS.items():
                if keyword in text:
                    total_score += weight
                    keywords_found[keyword] = weight
            for keyword, weight in POSITIVE_KEYWORDS.items():
                if keyword in text:
                    total_score = max(total_score - weight, 0)

        normaliser = max(len(articles), 1) * 2.0
        scaled_score = min(total_score / normaliser, 1.0)
        if scaled_score >= 0.66:
            severity = "High"
        elif scaled_score >= 0.33:
            severity = "Medium"
        else:
            severity = "Low"

        scores[country] = {
            "score": float(scaled_score),
            "severity": severity,
            "keywords": sorted(keywords_found.keys()),
            "article_count": len(articles),
        }
    return scores


def compute_supplier_risk_scores(
    supplier_metrics: pd.DataFrame,
    news_scores: Dict[str, Dict],
    weights: Dict[str, float] | None = None,
) -> Tuple[pd.DataFrame, List[SupplierRiskResult]]:
    """Compute risk scores for each supplier and provide explainability details."""

    if weights is None:
        weights = RISK_WEIGHTS

    metrics = supplier_metrics.copy()
    components: Dict[str, Dict[str, float]] = {}

    delivery_risk = _min_max_scale(metrics["avg_delivery_time"]).clip(0, 1)
    volatility_risk = _min_max_scale(metrics["price_volatility"]).clip(0, 1)
    spend_risk = _min_max_scale(metrics["spend_share"]).clip(0, 1)
    on_time_risk = (1 - metrics["on_time_rate"]).clip(0, 1)
    quality_risk = metrics["defect_rate"].clip(0, 1)

    news_component = metrics["country"].map(
        lambda c: news_scores.get(c, {}).get("score", 0.0)
    ).fillna(0)

    for idx, row in metrics.iterrows():
        supplier_components = {
            "Delivery performance": float(delivery_risk.loc[idx] * weights["delivery"] * 100),
            "On-time reliability": float(on_time_risk.loc[idx] * weights["on_time"] * 100),
            "Quality issues": float(quality_risk.loc[idx] * weights["quality"] * 100),
            "Price volatility": float(volatility_risk.loc[idx] * weights["volatility"] * 100),
            "Spend concentration": float(spend_risk.loc[idx] * weights["spend"] * 100),
            "Country news": float(news_component.loc[idx] * weights["news"] * 100),
        }
        components[row["supplier_id"]] = supplier_components

    metrics["risk_score"] = [sum(components[row["supplier_id"]].values()) for _, row in metrics.iterrows()]
    metrics["risk_level"] = metrics["risk_score"].apply(
        lambda score: "High" if score >= 66 else ("Medium" if score >= 33 else "Low")
    )
    metrics["top_factor"] = [
        max(comp.items(), key=lambda item: item[1])[0] if comp else "N/A"
        for comp in components.values()
    ]
    metrics["news_severity"] = metrics["country"].map(
        lambda c: news_scores.get(c, {}).get("severity", "Low")
    )

    results: List[SupplierRiskResult] = []
    for _, row in metrics.iterrows():
        metrics_dict = {
            "avg_delivery_time": float(row["avg_delivery_time"]),
            "on_time_rate": float(row["on_time_rate"]),
            "defect_rate": float(row["defect_rate"]),
            "price_volatility": float(row["price_volatility"]),
            "spend_share": float(row["spend_share"]),
            "news_score": float(news_scores.get(row["country"], {}).get("score", 0.0)),
        }
        result = SupplierRiskResult(
            supplier_id=row["supplier_id"],
            supplier_name=row["supplier_name"],
            country=row["country"],
            metrics=metrics_dict,
            components=components[row["supplier_id"]],
        )
        results.append(result)

    return metrics, results


def build_country_risk_table(news_scores: Dict[str, Dict]) -> pd.DataFrame:
    """Create a tidy summary table for country-level news risk."""

    rows = []
    for country, details in news_scores.items():
        rows.append(
            {
                "country": country,
                "news_risk_score": round(details.get("score", 0.0) * 100, 1),
                "news_severity": details.get("severity", "Low"),
                "keywords": ", ".join(details.get("keywords", [])) or "-",
                "article_count": details.get("article_count", 0),
            }
        )
    return pd.DataFrame(rows)



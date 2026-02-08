from typing import List, Dict, Optional, Tuple
from datetime import datetime
import math
import logging

from .financials_service import get_financial_metrics

logger = logging.getLogger(__name__)


def _as_float(v: Optional[float]) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _percent_normalize(v: Optional[float]) -> float:
    """Normalize a %-style value into roughly [-1, +1]. Accepts either 10 (means 10%) or 0.10."""
    if v is None:
        return 0.0
    v = float(v)
    if abs(v) > 2:  # likely expressed as percent like 10 => 10%
        v = v / 100.0
    # clamp to [-2,2] then scale
    v = max(min(v, 2.0), -2.0)
    return v / 2.0  # map to [-1,1]


def _ratio_normalize(v: Optional[float], cap: float = 5.0) -> float:
    """Normalize ratios (ROE, turnover, leverage, etc.) into [-1,1] with a cap."""
    if v is None:
        return 0.0
    v = float(v)
    # Favor positive values, penalize negative; cap extremes
    if v >= 0:
        return min(v / cap, 1.0)
    return -min(abs(v) / cap, 1.0)


def _money_normalize(v: Optional[float]) -> float:
    """Normalize monetary amounts to [-1,1] using log scaling (preserve sign)."""
    if v is None:
        return 0.0
    try:
        v = float(v)
    except Exception:
        return 0.0
    if v == 0:
        return 0.0
    sign = 1.0 if v > 0 else -1.0
    return sign * (math.log1p(abs(v)) / (1 + math.log1p(abs(v))))  # in (0,1)


def _contrib(name: str, value: Optional[float]) -> float:
    """Choose normalization based on metric name."""
    lname = name.lower()
    if "growth" in lname or "rate" in lname:
        return _percent_normalize(value)
    if "margin" in lname or "coverage" in lname or "turnover" in lname or "leverage" in lname or "debt" in lname or "roe" in lname:
        return _ratio_normalize(value)
    if "eps" in lname or "profit" in lname or "sales" in lname or "receivable" in lname or "inventory" in lname or "cash flow" in lname:
        return _money_normalize(value)
    # fallback
    return _ratio_normalize(value)


# weights for metrics (tuned heuristically)
_WEIGHTS = {
    "Sales Growth": 0.12,
    "Receivables Growth": 0.03,
    "Inventory Growth": 0.02,
    "Operating Profit": 0.05,
    "Net profit": 0.08,
    "Net profit Growth": 0.12,
    "EPS": 0.06,
    "EPS Growth": 0.10,
    "Operating Profit Margin": 0.08,
    "Net profit Margin": 0.06,
    "Asset Turnover": 0.04,
    "Financial Leverage": -0.06,
    "Return on Equity": 0.12,
    "Debt to equity ratio": -0.08,
    "Interest Coverage": 0.06,
    "Tax rate": -0.02,
    "Cash Flow from operation": 0.16,
    # fallback weight for any other metric (small)
    "__default__": 0.01,
}


def _score_from_metrics(metrics: Dict[str, Optional[float]]) -> Tuple[float, List[Tuple[str, float, float]]]:
    """
    Compute a composite score and list of contributions:
    returns (score, [(metric_name, raw_value, contribution), ...])
    """
    total = 0.0
    contributions = []
    for name, raw in metrics.items():
        weight = _WEIGHTS.get(name, _WEIGHTS["__default__"])
        val = _as_float(raw)
        norm = _contrib(name, val)
        contrib = weight * norm
        total += contrib
        contributions.append((name, val if val is not None else 0.0, contrib))
    return total, contributions


def _label_from_score(score: float) -> str:
    """Map numeric score to recommendation label."""
    if score >= 0.20:
        return "BUY"
    if score >= 0.00:
        return "HOLD"
    return "SELL"


def recommend(tickers: List[str], year: Optional[int] = None, top_k: int = 5) -> List[Dict]:
    """
    Produce simple rule-based recommendations for provided tickers.

    For each ticker:
      - fetch financial metrics for the target year (default: previous fiscal year)
      - compute a heuristic score combining growth, profitability, leverage and cash flow
      - return sorted recommendations by score descending (top_k)
    """
    results = []
    target_year = year if year is not None else (datetime.now().year - 1)

    for t in tickers:
        try:
            metrics = get_financial_metrics(t, target_year) or {}
        except Exception as e:
            logger.warning("Failed to fetch financials for %s: %s", t, e)
            metrics = {}

        score, contribs = _score_from_metrics(metrics)

        # sort contributions by absolute impact for short rationale
        contribs_sorted = sorted(contribs, key=lambda x: abs(x[2]), reverse=True)[:6]
        rationale = [
            {"metric": c[0], "value": c[1], "contribution": round(c[2], 4)} for c in contribs_sorted
        ]

        label = _label_from_score(score)

        results.append(
            {
                "ticker": t.upper(),
                "year": target_year,
                "score": round(score, 4),
                "recommendation": label,
                "rationale": rationale,
                "raw_metrics": metrics,
            }
        )

    # sort by score descending and return top_k
    results_sorted = sorted(results, key=lambda r: r["score"], reverse=True)
    return results_sorted[:max(1, min(top_k, len(results_sorted)))]

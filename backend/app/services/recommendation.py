from typing import List, Dict


def recommend(tickers: List[str], top_k: int = 5) -> List[Dict]:
    """Simple placeholder recommendation engine.

    Returns tickers (up to top_k) with dummy scores. Replace this with real model/service calls.
    """
    return [{"ticker": t, "score": float(top_k - i)} for i, t in enumerate(tickers[:top_k])]

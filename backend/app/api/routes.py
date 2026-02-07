from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()


class RecommendationRequest(BaseModel):
    tickers: List[str]
    top_k: int = 5


class RecommendationResponse(BaseModel):
    ticker: str
    score: float


@router.post("/recommend", response_model=List[RecommendationResponse])
async def recommend(req: RecommendationRequest):
    """Return dummy recommendations for the requested tickers.

    This placeholder uses simple deterministic scores. Replace with real logic in backend.app.services.recommendation.
    """
    # naive placeholder: give descending dummy scores
    results = []
    for i, t in enumerate(req.tickers[: req.top_k]):
        results.append({"ticker": t, "score": float(req.top_k - i)})
    return results

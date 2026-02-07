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
    # simple deterministic placeholder
    return [{"ticker": t, "score": float(req.top_k - i)} for i, t in enumerate(req.tickers[: req.top_k])]

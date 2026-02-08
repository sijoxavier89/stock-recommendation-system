from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

from ..services.recommender_service import get_recommendation
from ..services import embedding_service

router = APIRouter()


class RecommendRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol, e.g. AAPL")
    query: str = Field(..., description="User question or analysis request")
    years: Optional[List[int]] = Field(None, description="Explicit years to search, e.g. [2021,2022,2023]")
    last_n: Optional[int] = Field(3, description="If years not provided, search last N years")
    top_k: Optional[int] = Field(5, description="Top K results per collection to retrieve")


class RecommendResponse(BaseModel):
    ticker: str
    years_searched: List[int]
    retrieved_count: int
    sources: List[str]
    llm_answer: str
    retrieved_chunks: List[Dict[str, Any]]


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(req: RecommendRequest):
    try:
        # quick availability check to return a 503 when embeddings are not ready
        if embedding_service is None:
            raise RuntimeError("Embedding service unavailable — install sentence-transformers or configure embedding provider")
        result = get_recommendation(
            ticker=req.ticker,
            user_query=req.query,
            years=req.years,
            last_n=req.last_n,
            top_k=req.top_k or 5,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # RuntimeError used by service to signal missing or unavailable dependencies
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from ..services.financials_service import get_financial_metrics

router = APIRouter()

class FinancialsRequest(BaseModel):
    ticker: str
    year: int

@router.post("/financials", response_model=Dict[str, Any])
async def financials(req: FinancialsRequest):
    try:
        metrics = get_financial_metrics(req.ticker, req.year)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
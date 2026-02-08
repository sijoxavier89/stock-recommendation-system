from fastapi import APIRouter, HTTPException
from typing import Dict

from ..services.financials_service import get_financial_metrics

router = APIRouter()


@router.get("/")
async def list_companies():
    """Placeholder: return a small sample list of companies."""
    return [{"ticker": "AAPL", "name": "Apple Inc."}, {"ticker": "MSFT", "name": "Microsoft"}]


@router.get("/{ticker}/financials/{year}")
async def company_financials(ticker: str, year: int) -> Dict:
    """Return financial metrics for a company and year.

    Example: GET /companies/AAPL/financials/2022
    """
    try:
        metrics = get_financial_metrics(ticker, int(year))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return metrics

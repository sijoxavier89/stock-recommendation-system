from fastapi import APIRouter

router = APIRouter()


@router.get("/{ticker}")
async def get_news(ticker: str):
    """Placeholder news endpoint."""
    return {"ticker": ticker, "headlines": []}

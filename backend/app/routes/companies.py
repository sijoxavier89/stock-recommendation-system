from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_companies():
    """Placeholder: return a small sample list of companies."""
    return [{"ticker": "AAPL", "name": "Apple Inc."}, {"ticker": "MSFT", "name": "Microsoft"}]

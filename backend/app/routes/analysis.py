from fastapi import APIRouter

router = APIRouter()


@router.post("/")
async def analyze(payload: dict):
    """Placeholder analysis endpoint."""
    return {"query": payload}

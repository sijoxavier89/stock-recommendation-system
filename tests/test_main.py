import pytest
from httpx import AsyncClient

from backend.app.main import app


@pytest.mark.asyncio
async def test_recommend_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        payload = {"tickers": ["AAPL", "MSFT", "GOOG"], "top_k": 2}
        r = await ac.post("/api/recommend", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["ticker"] == "AAPL"

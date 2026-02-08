import pytest
from httpx import AsyncClient

from backend.app.main import app


@pytest.mark.asyncio
async def test_recommend_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        payload = {"ticker": "AAPL", "query": "How did revenue trend over the last 3 years?", "last_n": 3, "top_k": 2}
        r = await ac.post("/api/recommend", json=payload)
        assert r.status_code == 200, r.text
        data = r.json()
        # Basic shape checks
        assert "ticker" in data
        assert data["ticker"] == "AAPL"
        assert "llm_answer" in data
        assert "retrieved_chunks" in data and isinstance(data["retrieved_chunks"], list)

import pytest
from httpx import AsyncClient
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from backend.app.main import app


@pytest.mark.asyncio
async def test_upload_triggers_pipeline(tmp_path, monkeypatch):
    """Upload a tiny PDF and verify the ingestion pipeline is invoked and file is saved."""
    # create tiny valid-ish PDF bytes (starts with %PDF)
    pdf_path = tmp_path / "sample.pdf"
    pdf_bytes = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n%%EOF\n"
    pdf_path.write_bytes(pdf_bytes)

    # Monkeypatch the pipeline singleton to a simple mock object
    mock_pipeline = SimpleNamespace(ingest_pdf=Mock())
    import backend.app.services as services
    import backend.app.routes.upload as upload_mod
    # patch both the services singleton and the already-imported symbol in the upload module
    monkeypatch.setattr(services, "pipeline", mock_pipeline)
    monkeypatch.setattr(upload_mod, "pipeline", mock_pipeline)

    # Perform the upload using AsyncClient against the FastAPI app
    async with AsyncClient(app=app, base_url="http://test") as ac:
        files = {"file": ("sample.pdf", pdf_bytes, "application/pdf")}
        data = {"ticker": "MSFT", "year": "2025"}
        resp = await ac.post("/api/upload/", files=files, data=data)

    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["ticker"] == "MSFT"

    # saved file should exist
    saved = Path(body["path"])
    assert saved.exists()

    # Ensure pipeline.ingest_pdf was scheduled/called
    assert mock_pipeline.ingest_pdf.called, "Expected pipeline.ingest_pdf to be called"

    # cleanup
    try:
        saved.unlink()
    except Exception:
        pass

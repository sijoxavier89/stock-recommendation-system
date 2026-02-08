from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks
from pathlib import Path
from uuid import uuid4
import logging

# replace manual path calculation with backend config
from ..core.config import ANNUAL_REPORTS_DIR as UPLOAD_DIR

logger = logging.getLogger(__name__)

# ensure upload dir exists (config already created it, keep idempotent)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Expose pipeline at module level so tests can monkeypatch it.
# Import in a try/except to avoid hard import-time failures when pipeline isn't ready.
try:
    from backend.app.services import pipeline
except Exception:
    pipeline = None

router = APIRouter()


@router.post("/", status_code=201)
async def upload_report(
    file: UploadFile = File(...),
    ticker: str = Form(...),
    year: int = Form(...),
    background_tasks: BackgroundTasks = None,
):
    """Accept a PDF upload along with `ticker` (stock symbol) and `year` (fiscal year).

    The file will be saved under `backend/data/annual_reports/{TICKER}/{year}_{filename}`.
    Requires `python-multipart` for form parsing.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    # sanitize ticker and filename
    ticker_safe = ticker.strip().upper()
    safe_name = Path(file.filename).name
    # simple sanitize: keep alnum, dash, underscore, dot
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in ("-", "_", ".")) or "upload.pdf"
    unique_prefix = uuid4().hex[:8]
    dest_dir = UPLOAD_DIR / ticker_safe
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{year}_{unique_prefix}_{safe_name}"

    # stream write to avoid OOM
    try:
        with dest_path.open("wb") as f:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                raise HTTPException(status_code=400, detail="Empty file")
            # basic magic header check for PDF
            if not chunk.startswith(b"%PDF"):
                # read a bit more in case header split
                rest = await file.read(1024)
                chunk = chunk + rest
                if not chunk.startswith(b"%PDF"):
                    raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF")
            f.write(chunk)
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed saving upload: %s", e)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # schedule ingestion pipeline if available
    from ..services import pipeline  # import here to avoid cycle at module import
    if background_tasks is not None and getattr(pipeline, "ingest_pdf", None):
        background_tasks.add_task(pipeline.ingest_pdf, str(dest_path), ticker_safe, int(year))
    else:
        # try best-effort synchronous call (fallback)
        try:
            if getattr(pipeline, "ingest_pdf", None):
                pipeline.ingest_pdf(str(dest_path), ticker_safe, int(year))
            else:
                logger.info("Ingestion pipeline not available; skipping indexing for %s", dest_path)
        except Exception:
            logger.exception("Background ingestion failed for %s", dest_path)

    return {"filename": file.filename, "ticker": ticker_safe, "year": int(year), "path": str(dest_path)}

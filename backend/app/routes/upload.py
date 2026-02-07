from fastapi import APIRouter, UploadFile, File, HTTPException
import os
from pathlib import Path

router = APIRouter()

UPLOAD_DIR = Path(__file__).resolve().parents[3] / "data" / "annual_reports"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/", status_code=201)
async def upload_report(file: UploadFile = File(...)):
    """Accept a PDF upload and save it to `backend/data/annual_reports/`.

    This endpoint requires `python-multipart`.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Basic validation: ensure a PDF
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are accepted")

    dest_path = UPLOAD_DIR / file.filename
    # write file to disk
    try:
        with dest_path.open("wb") as f:
            content = await file.read()
            f.write(content)
    finally:
        await file.close()

    return {"filename": file.filename, "path": str(dest_path)}

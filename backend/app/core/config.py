from pathlib import Path

# backend/ (two parents up from this file)
ROOT = Path(__file__).resolve().parents[2]

# runtime storage under backend/
DATA_DIR = ROOT / "data"
ANNUAL_REPORTS_DIR = DATA_DIR / "annual_reports"
CHROMA_DB_DIR = ROOT / "chroma_db"

# ensure dirs exist
ANNUAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

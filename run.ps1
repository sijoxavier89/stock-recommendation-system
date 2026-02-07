# PowerShell helper to run the app locally
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload

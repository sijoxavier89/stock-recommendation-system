from fastapi import FastAPI
from .routes import router as api_router
from app.routes import recommendation  # add this


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="Stock Recommendation System")
    app.include_router(api_router, prefix="/api")
    app.include_router(recommendation.router, prefix="/api", tags=["recommendation"])
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)

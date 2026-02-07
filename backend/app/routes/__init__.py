from fastapi import APIRouter

from .companies import router as companies_router
from .upload import router as upload_router
from .analysis import router as analysis_router
from .news import router as news_router
from .recommendation import router as recommendation_router

router = APIRouter()

# mount sub-routers
router.include_router(companies_router, prefix="/companies")
router.include_router(upload_router, prefix="/upload")
router.include_router(analysis_router, prefix="/analysis")
router.include_router(news_router, prefix="/news")
router.include_router(recommendation_router, prefix="")

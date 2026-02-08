"""
Analysis API Routes
Endpoints for querying and analyzing company data using RAG.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.services.rag_service import RAGService, RAGResponse
from app.dependencies import get_rag_service

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


# Request/Response Models

class AnalysisRequest(BaseModel):
    """Request for general analysis query."""
    query: str = Field(..., description="Natural language question about company/companies")
    company_ticker: Optional[str] = Field(None, description="Optional ticker to filter results")
    year: Optional[int] = Field(None, description="Optional year to filter results")
    section_type: Optional[str] = Field(None, description="Optional section type filter")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What was Apple's revenue growth in 2023?",
                "company_ticker": "AAPL",
                "year": 2023
            }
        }


class ComparisonRequest(BaseModel):
    """Request for comparing multiple companies."""
    company_tickers: List[str] = Field(..., min_length=2, max_length=5, description="2-5 company tickers to compare")
    metric_focus: Optional[str] = Field(None, description="Optional metric to focus on (e.g., 'ROE', 'growth')")
    year: Optional[int] = Field(None, description="Optional year for comparison")
    
    class Config:
        json_schema_extra = {
            "example": {
                "company_tickers": ["AAPL", "MSFT", "GOOGL"],
                "metric_focus": "ROE",
                "year": 2023
            }
        }


class RecommendationRequest(BaseModel):
    """Request for investment recommendation."""
    company_ticker: str = Field(..., description="Company ticker symbol")
    investment_style: str = Field(
        "balanced",
        description="Investment style: 'growth', 'value', or 'balanced'"
    )
    year: Optional[int] = Field(None, description="Optional year to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "company_ticker": "AAPL",
                "investment_style": "growth",
                "year": 2023
            }
        }


class RetrievedChunk(BaseModel):
    """A retrieved document chunk with metadata."""
    text: str
    company_name: str
    ticker: str
    year: int
    section_type: str
    similarity_score: float
    page_numbers: List[int]


class AnalysisResponse(BaseModel):
    """Response from analysis endpoint."""
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    metrics_used: Optional[Dict[str, Any]] = None
    confidence: float
    reasoning: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# API Endpoints

@router.post("/query", response_model=AnalysisResponse)
async def analyze_query(
    request: AnalysisRequest,
    rag: RAGService = Depends(get_rag_service)
) -> AnalysisResponse:
    """
    Answer a natural language query about companies using RAG.
    
    This endpoint:
    1. Retrieves relevant document chunks from vector store
    2. Extracts financial metrics from chunks
    3. Uses LLM to generate evidence-based answer
    4. Returns answer with sources and confidence score
    
    Example queries:
    - "What was Apple's revenue in 2023?"
    - "How did Microsoft's profit margins change over time?"
    - "What are the main risks for Tesla?"
    """
    try:
        # Execute RAG query
        rag_response = rag.query(
            user_query=request.query,
            company_filter=request.company_ticker,
            year_filter=request.year,
            section_filter=request.section_type
        )
        
        # Convert to API response format
        chunks = [
            RetrievedChunk(
                text=chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
                company_name=chunk.metadata.get("company_name", "Unknown"),
                ticker=chunk.metadata.get("ticker", "N/A"),
                year=chunk.metadata.get("year", 0),
                section_type=chunk.metadata.get("section_type", "general"),
                similarity_score=chunk.similarity_score,
                page_numbers=chunk.metadata.get("page_numbers", [])
                    if isinstance(chunk.metadata.get("page_numbers"), list)
                    else [int(p) for p in str(chunk.metadata.get("page_numbers", "")).split(",") if p]
            )
            for chunk in rag_response.retrieved_chunks
        ]
        
        return AnalysisResponse(
            answer=rag_response.answer,
            retrieved_chunks=chunks,
            metrics_used=rag_response.metrics_used,
            confidence=rag_response.confidence,
            reasoning=rag_response.reasoning
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/compare", response_model=AnalysisResponse)
async def compare_companies(
    request: ComparisonRequest,
    rag: RAGService = Depends(get_rag_service)
) -> AnalysisResponse:
    """
    Compare multiple companies across key metrics.
    
    This endpoint:
    1. Retrieves data for all specified companies
    2. Extracts and aggregates metrics
    3. Generates comparative analysis using LLM
    4. Highlights relative strengths/weaknesses
    
    Example:
    - Compare AAPL, MSFT, GOOGL on ROE
    - Compare TSLA vs F on growth metrics
    """
    try:
        # Validate tickers are uppercase
        tickers = [t.upper() for t in request.company_tickers]
        
        # Execute comparison
        rag_response = rag.compare_companies(
            company_tickers=tickers,
            metric_focus=request.metric_focus,
            year=request.year
        )
        
        # Convert to API response
        chunks = [
            RetrievedChunk(
                text=chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
                company_name=chunk.metadata.get("company_name", "Unknown"),
                ticker=chunk.metadata.get("ticker", "N/A"),
                year=chunk.metadata.get("year", 0),
                section_type=chunk.metadata.get("section_type", "general"),
                similarity_score=chunk.similarity_score,
                page_numbers=chunk.metadata.get("page_numbers", [])
                    if isinstance(chunk.metadata.get("page_numbers"), list)
                    else [int(p) for p in str(chunk.metadata.get("page_numbers", "")).split(",") if p]
            )
            for chunk in rag_response.retrieved_chunks
        ]
        
        return AnalysisResponse(
            answer=rag_response.answer,
            retrieved_chunks=chunks,
            metrics_used=rag_response.metrics_used,
            confidence=rag_response.confidence,
            reasoning=rag_response.reasoning
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}"
        )


@router.post("/recommend", response_model=AnalysisResponse)
async def get_recommendation(
    request: RecommendationRequest,
    rag: RAGService = Depends(get_rag_service)
) -> AnalysisResponse:
    """
    Get investment recommendation for a company.
    
    This endpoint:
    1. Analyzes comprehensive financial data
    2. Evaluates against investment criteria
    3. Considers growth, profitability, and risks
    4. Provides BUY/HOLD/AVOID recommendation with reasoning
    
    Investment styles:
    - **growth**: Focus on revenue/EPS growth, expanding margins
    - **value**: Focus on profitability, low debt, stable cash flow
    - **balanced**: Moderate growth with good fundamentals
    
    Note: This is for educational purposes only, not financial advice.
    """
    try:
        # Validate investment style
        valid_styles = ["growth", "value", "balanced"]
        if request.investment_style.lower() not in valid_styles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid investment style. Must be one of: {', '.join(valid_styles)}"
            )
        
        # Validate ticker format
        ticker = request.company_ticker.upper()
        
        # Generate recommendation
        rag_response = rag.get_recommendation(
            company_ticker=ticker,
            investment_style=request.investment_style.lower(),
            year=request.year
        )
        
        # Convert to API response
        chunks = [
            RetrievedChunk(
                text=chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
                company_name=chunk.metadata.get("company_name", "Unknown"),
                ticker=chunk.metadata.get("ticker", "N/A"),
                year=chunk.metadata.get("year", 0),
                section_type=chunk.metadata.get("section_type", "general"),
                similarity_score=chunk.similarity_score,
                page_numbers=chunk.metadata.get("page_numbers", [])
                    if isinstance(chunk.metadata.get("page_numbers"), list)
                    else [int(p) for p in str(chunk.metadata.get("page_numbers", "")).split(",") if p]
            )
            for chunk in rag_response.retrieved_chunks
        ]
        
        return AnalysisResponse(
            answer=rag_response.answer,
            retrieved_chunks=chunks,
            metrics_used=rag_response.metrics_used,
            confidence=rag_response.confidence,
            reasoning=rag_response.reasoning
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {str(e)}"
        )


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "analysis",
        "timestamp": datetime.now().isoformat()
    }

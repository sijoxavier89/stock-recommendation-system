"""
RAG Service
Retrieval-Augmented Generation for stock analysis and recommendations.
Combines vector search with LLM reasoning.
"""
from typing import List, Dict, Optional, Any
import logging
from dataclasses import dataclass
import json

from .vector_store import VectorStore
from .embedding_service import EmbeddingService
from .llm_service import LLMService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieved document chunk with metadata."""
    text: str
    metadata: Dict[str, Any]
    similarity_score: float
    chunk_id: str


@dataclass
class RAGResponse:
    """Complete RAG response with answer and sources."""
    answer: str
    retrieved_chunks: List[RetrievalResult]
    company_context: Optional[Dict] = None
    metrics_used: Optional[Dict] = None
    confidence: float = 0.0
    reasoning: Optional[str] = None


class RAGService:
    """
    Retrieval-Augmented Generation service for stock recommendations.
    
    Flow:
    1. User Query → Parse intent (company, metrics, timeframe)
    2. Retrieve relevant chunks from vector store
    3. Extract metrics from chunks
    4. Build context with chunks + metrics
    5. LLM generates answer with reasoning
    6. Return structured response
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        llm_service: LLMService,
        top_k: int = 5
    ):
        """
        Initialize RAG service.
        
        Args:
            vector_store: Vector database for retrieval
            embedding_service: Embedding generator
            llm_service: LLM for generation
            top_k: Number of chunks to retrieve
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.top_k = top_k
        
        logger.info("RAG service initialized")
    
    def query(
        self,
        user_query: str,
        company_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
        section_filter: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> RAGResponse:
        """
        Answer a user query using RAG.
        
        Args:
            user_query: Natural language question
            company_filter: Ticker symbol to filter by
            year_filter: Year to filter by
            section_filter: Section type to filter by
            top_k: Override default number of chunks
        
        Returns:
            RAGResponse with answer and sources
        """
        logger.info(f"Processing query: {user_query}")
        
        # Step 1: Parse query intent (optional but helpful)
        intent = self._parse_query_intent(user_query)
        logger.info(f"Detected intent: {intent}")
        
        # Step 2: Retrieve relevant chunks
        retrieved_chunks = self._retrieve_relevant_chunks(
            query=user_query,
            company_filter=company_filter or intent.get("company"),
            year_filter=year_filter or intent.get("year"),
            section_filter=section_filter or intent.get("section_type"),
            top_k=top_k or self.top_k
        )
        
        if not retrieved_chunks:
            return RAGResponse(
                answer="I couldn't find relevant information to answer your question. Please try rephrasing or check if the company data is available.",
                retrieved_chunks=[],
                confidence=0.0
            )
        
        # Step 3: Extract metrics from chunks
        metrics = self._extract_metrics_from_chunks(retrieved_chunks)
        
        # Step 4: Build context for LLM
        context = self._build_context(
            query=user_query,
            chunks=retrieved_chunks,
            metrics=metrics,
            intent=intent
        )
        
        # Step 5: Generate answer using LLM
        answer, reasoning = self._generate_answer(
            query=user_query,
            context=context,
            intent=intent
        )
        
        # Step 6: Calculate confidence
        confidence = self._calculate_confidence(retrieved_chunks, answer)
        
        return RAGResponse(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            metrics_used=metrics,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def compare_companies(
        self,
        company_tickers: List[str],
        metric_focus: Optional[str] = None,
        year: Optional[int] = None
    ) -> RAGResponse:
        """
        Compare multiple companies across key metrics.
        
        Args:
            company_tickers: List of ticker symbols
            metric_focus: Optional metric to focus on (e.g., "ROE", "growth")
            year: Optional year to compare
        
        Returns:
            RAGResponse with comparative analysis
        """
        logger.info(f"Comparing companies: {company_tickers}")
        
        # Retrieve data for each company
        company_data = {}
        for ticker in company_tickers:
            chunks = self._retrieve_relevant_chunks(
                query=metric_focus or "financial highlights",
                company_filter=ticker,
                year_filter=year,
                top_k=10
            )
            metrics = self._extract_metrics_from_chunks(chunks)
            
            company_data[ticker] = {
                "chunks": chunks,
                "metrics": metrics
            }
        
        # Build comparative context
        context = self._build_comparative_context(company_data, metric_focus)
        
        # Generate comparison
        query = f"Compare {', '.join(company_tickers)}"
        if metric_focus:
            query += f" focusing on {metric_focus}"
        
        answer, reasoning = self._generate_answer(
            query=query,
            context=context,
            intent={"type": "comparison", "companies": company_tickers}
        )
        
        # Combine all chunks
        all_chunks = []
        for data in company_data.values():
            all_chunks.extend(data["chunks"])
        
        return RAGResponse(
            answer=answer,
            retrieved_chunks=all_chunks,
            company_context=company_data,
            reasoning=reasoning,
            confidence=0.85
        )
    
    def get_recommendation(
        self,
        company_ticker: str,
        investment_style: str = "growth",  # "growth", "value", "balanced"
        year: Optional[int] = None
    ) -> RAGResponse:
        """
        Get investment recommendation for a company.
        
        Args:
            company_ticker: Company ticker symbol
            investment_style: Investment approach
            year: Optional year to analyze
        
        Returns:
            RAGResponse with recommendation
        """
        logger.info(f"Generating recommendation for {company_ticker} ({investment_style} style)")
        
        # Retrieve comprehensive data
        chunks = self._retrieve_relevant_chunks(
            query="financial performance metrics growth profitability",
            company_filter=company_ticker,
            year_filter=year,
            top_k=15
        )
        
        metrics = self._extract_metrics_from_chunks(chunks)
        
        # Build recommendation context
        context = self._build_recommendation_context(
            chunks=chunks,
            metrics=metrics,
            investment_style=investment_style
        )
        
        # Generate recommendation
        query = f"Should I invest in {company_ticker}? I prefer {investment_style} stocks."
        answer, reasoning = self._generate_answer(
            query=query,
            context=context,
            intent={"type": "recommendation", "style": investment_style}
        )
        
        return RAGResponse(
            answer=answer,
            retrieved_chunks=chunks,
            metrics_used=metrics,
            reasoning=reasoning,
            confidence=self._calculate_confidence(chunks, answer)
        )
    
    def _parse_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Parse user query to extract intent (company, metrics, timeframe).
        Uses simple heuristics + optional LLM call.
        """
        intent = {
            "type": "general",
            "company": None,
            "year": None,
            "section_type": None,
            "metrics": []
        }
        
        query_lower = query.lower()
        
        # Detect query type
        if any(word in query_lower for word in ["compare", "vs", "versus", "better"]):
            intent["type"] = "comparison"
        elif any(word in query_lower for word in ["recommend", "should i buy", "invest"]):
            intent["type"] = "recommendation"
        elif any(word in query_lower for word in ["trend", "over time", "history"]):
            intent["type"] = "trend"
        
        # Detect section focus
        if any(word in query_lower for word in ["risk", "risks"]):
            intent["section_type"] = "risk_factors"
        elif any(word in query_lower for word in ["revenue", "sales", "profit", "earnings"]):
            intent["section_type"] = "financial_highlights"
        
        # Detect year (simple pattern)
        import re
        year_match = re.search(r'\b(20\d{2})\b', query)
        if year_match:
            intent["year"] = int(year_match.group(1))
        
        # Detect specific metrics mentioned
        metric_keywords = {
            "revenue": "sales",
            "sales": "sales",
            "profit": "net_profit",
            "earnings": "net_profit",
            "eps": "eps",
            "margin": "net_profit_margin",
            "roe": "return_on_equity",
            "debt": "debt_to_equity",
            "cash flow": "cash_flow_from_operations",
        }
        
        for keyword, metric in metric_keywords.items():
            if keyword in query_lower:
                intent["metrics"].append(metric)
        
        return intent
    
    def _retrieve_relevant_chunks(
        self,
        query: str,
        company_filter: Optional[str] = None,
        year_filter: Optional[int] = None,
        section_filter: Optional[str] = None,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve most relevant chunks from vector store.
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_text(query)
        
        # Build filter conditions
        where = {}
        if company_filter:
            where["ticker"] = company_filter.upper()
        if year_filter:
            where["year"] = year_filter
        if section_filter:
            where["section_type"] = section_filter
        
        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            n_results=top_k,
            where=where if where else None
        )
        
        # Convert to RetrievalResult objects
        chunks = []
        for i in range(len(results["documents"])):
            chunk = RetrievalResult(
                text=results["documents"][i],
                metadata=results["metadatas"][i],
                similarity_score=1.0 - results["distances"][i],  # Convert distance to similarity
                chunk_id=results["ids"][i]
            )
            chunks.append(chunk)
        
        logger.info(f"Retrieved {len(chunks)} chunks (avg similarity: {sum(c.similarity_score for c in chunks) / len(chunks):.3f})")
        
        return chunks
    
    def _extract_metrics_from_chunks(
        self,
        chunks: List[RetrievalResult]
    ) -> Dict[str, Any]:
        """
        Extract and aggregate financial metrics from retrieved chunks.
        """
        metrics = {}
        
        for chunk in chunks:
            chunk_metrics = chunk.metadata.get("metrics")
            
            if isinstance(chunk_metrics, str):
                # Parse JSON string if needed
                try:
                    chunk_metrics = json.loads(chunk_metrics)
                except:
                    continue
            
            if isinstance(chunk_metrics, dict):
                # Merge metrics, preferring higher confidence values
                for metric_name, value in chunk_metrics.items():
                    if metric_name not in metrics:
                        metrics[metric_name] = value
        
        logger.info(f"Extracted {len(metrics)} unique metrics from chunks")
        return metrics
    
    def _build_context(
        self,
        query: str,
        chunks: List[RetrievalResult],
        metrics: Dict[str, Any],
        intent: Dict[str, Any]
    ) -> str:
        """
        Build context string for LLM from retrieved chunks and metrics.
        """
        context_parts = []
        
        # Add company information
        if chunks:
            company = chunks[0].metadata.get("company_name", "Unknown")
            ticker = chunks[0].metadata.get("ticker", "N/A")
            year = chunks[0].metadata.get("year", "N/A")
            
            context_parts.append(f"Company: {company} ({ticker})")
            context_parts.append(f"Fiscal Year: {year}")
            context_parts.append("")
        
        # Add extracted metrics if available
        if metrics:
            context_parts.append("KEY FINANCIAL METRICS:")
            
            # Priority metrics to show
            priority_metrics = [
                ("sales", "Revenue"),
                ("sales_growth", "Revenue Growth"),
                ("net_profit", "Net Profit"),
                ("net_profit_margin", "Net Profit Margin"),
                ("eps", "EPS"),
                ("return_on_equity", "ROE"),
                ("debt_to_equity", "Debt/Equity"),
                ("cash_flow_from_operations", "Operating Cash Flow"),
            ]
            
            for metric_key, display_name in priority_metrics:
                if metric_key in metrics:
                    value = metrics[metric_key]
                    
                    # Format value
                    if metric_key in ["sales", "net_profit", "cash_flow_from_operations"]:
                        formatted = f"${value:,.0f}" if value >= 1_000_000 else f"${value:,.2f}"
                    elif metric_key in ["sales_growth", "net_profit_margin", "return_on_equity"]:
                        formatted = f"{value:.1f}%"
                    else:
                        formatted = f"{value:.2f}"
                    
                    context_parts.append(f"  {display_name}: {formatted}")
            
            context_parts.append("")
        
        # Add relevant document chunks
        context_parts.append("RELEVANT EXCERPTS FROM ANNUAL REPORT:")
        context_parts.append("")
        
        for i, chunk in enumerate(chunks[:5], 1):  # Limit to top 5
            section = chunk.metadata.get("section_type", "general")
            pages = chunk.metadata.get("page_numbers", "N/A")
            
            context_parts.append(f"[{i}] Section: {section}, Pages: {pages}")
            context_parts.append(chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _build_comparative_context(
        self,
        company_data: Dict[str, Dict],
        metric_focus: Optional[str] = None
    ) -> str:
        """
        Build context for comparing multiple companies.
        """
        context_parts = ["COMPANY COMPARISON:", ""]
        
        for ticker, data in company_data.items():
            metrics = data["metrics"]
            
            context_parts.append(f"--- {ticker} ---")
            
            if metrics:
                for metric_name, value in metrics.items():
                    if metric_focus and metric_focus.lower() not in metric_name.lower():
                        continue
                    context_parts.append(f"  {metric_name}: {value}")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _build_recommendation_context(
        self,
        chunks: List[RetrievalResult],
        metrics: Dict[str, Any],
        investment_style: str
    ) -> str:
        """
        Build context specifically for investment recommendations.
        """
        context = self._build_context(
            query="",
            chunks=chunks,
            metrics=metrics,
            intent={"type": "recommendation"}
        )
        
        # Add investment criteria based on style
        criteria_map = {
            "growth": [
                "High revenue growth (>15%)",
                "High EPS growth (>20%)",
                "Strong ROE (>20%)",
                "Expanding margins"
            ],
            "value": [
                "Strong profitability (Net margin >10%)",
                "Low debt (D/E <1.5)",
                "Positive cash flow",
                "Stable or growing dividends"
            ],
            "balanced": [
                "Moderate growth (>10%)",
                "Good profitability (ROE >15%)",
                "Reasonable debt levels (D/E <2.0)",
                "Positive cash generation"
            ]
        }
        
        criteria = criteria_map.get(investment_style, criteria_map["balanced"])
        
        context += f"\n\nINVESTMENT CRITERIA ({investment_style.upper()} STYLE):\n"
        context += "\n".join(f"  • {c}" for c in criteria)
        
        return context
    
    def _generate_answer(
        self,
        query: str,
        context: str,
        intent: Dict[str, Any]
    ) -> tuple[str, str]:
        """
        Generate answer using LLM with retrieved context.
        
        Returns:
            (answer, reasoning) tuple
        """
        # Build system prompt based on intent
        system_prompt = self._build_system_prompt(intent)
        
        # Build user prompt
        user_prompt = f"""Context information from annual reports:

{context}

User Question: {query}

Please provide a clear, evidence-based answer using the information above. Be specific and cite the metrics when relevant. If making a recommendation, explain your reasoning."""
        
        # Get response from LLM
        response = self.llm_service.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,  # Lower temperature for factual responses
            max_tokens=800
        )
        
        # Extract answer and reasoning
        answer = response.get("answer", "")
        reasoning = response.get("reasoning", "")
        
        return answer, reasoning
    
    def _build_system_prompt(self, intent: Dict[str, Any]) -> str:
        """
        Build system prompt based on query intent.
        """
        base_prompt = """You are a financial analyst AI assistant specializing in stock analysis and investment recommendations.

Your responsibilities:
1. Analyze financial data from annual reports
2. Provide evidence-based insights
3. Cite specific metrics and page numbers when available
4. Be honest about limitations and missing data
5. Never make up numbers or facts

Important guidelines:
- Always base your analysis on the provided context
- Use exact metrics from the reports
- Highlight both positives and risks
- If data is missing or unclear, say so
- Format numbers clearly (e.g., $394.3B, 25.3%)

Remember: This is for educational purposes only, not financial advice."""
        
        # Add intent-specific guidance
        intent_type = intent.get("type", "general")
        
        if intent_type == "recommendation":
            base_prompt += """

For investment recommendations:
1. Evaluate financial health (profitability, margins, debt)
2. Assess growth trajectory
3. Consider risks mentioned in the report
4. Provide a clear recommendation: BUY, HOLD, or AVOID
5. Explain your reasoning with specific metrics"""
        
        elif intent_type == "comparison":
            base_prompt += """

For company comparisons:
1. Compare key metrics side-by-side
2. Highlight relative strengths and weaknesses
3. Consider both quantitative and qualitative factors
4. Be balanced and objective"""
        
        elif intent_type == "trend":
            base_prompt += """

For trend analysis:
1. Show changes over time
2. Calculate growth rates
3. Identify patterns or inflection points
4. Explain what's driving the trends"""
        
        return base_prompt
    
    def _calculate_confidence(
        self,
        chunks: List[RetrievalResult],
        answer: str
    ) -> float:
        """
        Calculate confidence score for the response.
        
        Based on:
        - Number of chunks retrieved
        - Average similarity score
        - Answer length and completeness
        """
        if not chunks:
            return 0.0
        
        # Factor 1: Retrieval quality
        avg_similarity = sum(c.similarity_score for c in chunks) / len(chunks)
        retrieval_score = min(avg_similarity * 1.2, 1.0)
        
        # Factor 2: Number of sources
        source_score = min(len(chunks) / 5.0, 1.0)
        
        # Factor 3: Answer completeness
        answer_score = min(len(answer) / 300, 1.0) if answer else 0.0
        
        # Weighted average
        confidence = (
            retrieval_score * 0.5 +
            source_score * 0.3 +
            answer_score * 0.2
        )
        
        return round(confidence, 2)


# Example usage
if __name__ == "__main__":
    from .vector_store import VectorStore
    from .embedding_service import EmbeddingService
    from .llm_service import LLMService
    
    # Initialize services
    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    llm_service = LLMService()
    
    # Create RAG service
    rag = RAGService(
        vector_store=vector_store,
        embedding_service=embedding_service,
        llm_service=llm_service
    )
    
    # Example queries
    
    # 1. Simple query
    response = rag.query(
        "What was Apple's revenue growth in 2023?",
        company_filter="AAPL"
    )
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence}")
    
    # 2. Comparison
    response = rag.compare_companies(
        company_tickers=["AAPL", "MSFT"],
        metric_focus="ROE",
        year=2023
    )
    print(f"Comparison: {response.answer}")
    
    # 3. Recommendation
    response = rag.get_recommendation(
        company_ticker="AAPL",
        investment_style="growth"
    )
    print(f"Recommendation: {response.answer}")
    print(f"Reasoning: {response.reasoning}")

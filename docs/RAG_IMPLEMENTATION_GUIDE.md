# RAG Recommendation Flow Implementation Guide

## Overview

This guide explains how to implement and use the complete **Retrieval-Augmented Generation (RAG)** system for stock recommendations. The RAG flow combines vector search with LLM reasoning to provide evidence-based analysis.

---

## Architecture: How RAG Works

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ 1. QUERY UNDERSTANDING                                  │
│    Parse intent: comparison, recommendation, trend      │
│    Extract entities: companies, metrics, years          │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 2. RETRIEVAL (Vector Search)                            │
│    • Embed user query                                   │
│    • Search vector DB with filters                      │
│    • Retrieve top-k relevant chunks                     │
│    • Extract metrics from chunks                        │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 3. CONTEXT BUILDING                                     │
│    • Combine retrieved text chunks                      │
│    • Add structured metrics                             │
│    • Include company metadata                           │
│    • Format for LLM consumption                         │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 4. GENERATION (LLM)                                     │
│    • System prompt (role, guidelines)                   │
│    • User prompt (query + context)                      │
│    • Generate evidence-based answer                     │
│    • Extract reasoning                                  │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│ 5. RESPONSE FORMATTING                                  │
│    • Answer text                                        │
│    • Source citations                                   │
│    • Confidence score                                   │
│    • Metrics used                                       │
└─────────────────────────────────────────────────────────┘
```

---

## File Structure

Place these files in your backend:

```
backend/app/services/
├── rag_service.py              # ← Core RAG orchestration
├── llm_service.py              # ← LLM provider interface
├── vector_store.py             # ← Already implemented
├── embedding_service.py        # ← Already implemented
└── pdf_processor.py            # ← Already implemented

backend/app/routes/
└── analysis.py                 # ← API endpoints

backend/app/
└── dependencies.py             # ← Service initialization
```

---

## Setup Instructions

### 1. Install Additional Dependencies

Add to `requirements.txt`:

```txt
# Already in requirements.txt:
# fastapi, uvicorn, chromadb, sentence-transformers

# Add these:
openai==1.3.7              # If using OpenAI
anthropic==0.7.8           # If using Claude
requests==2.31.0           # For Ollama HTTP calls
```

Install:
```bash
pip install -r requirements.txt
```

### 2. Install Ollama (Local LLM - Free)

**For Mac/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**For Windows:**
Download from https://ollama.com/download

**Start Ollama:**
```bash
ollama serve
```

**Pull a model:**
```bash
ollama pull mistral  # Fast, good quality
# or
ollama pull llama3   # Larger, better quality
```

### 3. Configure Environment Variables

Update `.env`:

```ini
# LLM Configuration
LLM_PROVIDER=ollama              # "ollama", "openai", or "anthropic"
LLM_MODEL=mistral                # Model name
OLLAMA_BASE_URL=http://localhost:11434

# If using OpenAI:
# LLM_PROVIDER=openai
# LLM_MODEL=gpt-4o-mini
# OPENAI_API_KEY=sk-...

# If using Anthropic:
# LLM_PROVIDER=anthropic
# LLM_MODEL=claude-3-5-sonnet-20241022
# ANTHROPIC_API_KEY=sk-ant-...

# Embedding Model (already configured)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Store (already configured)
CHROMA_DB_PATH=./data/chroma_db
```

---

## Testing the RAG Flow

### Step 1: Test LLM Service Standalone

Create `backend/tests/test_llm.py`:

```python
from app.services.llm_service import LLMService

# Test Ollama
llm = LLMService(provider="ollama", model="mistral")

response = llm.generate(
    system_prompt="You are a financial analyst.",
    user_prompt="Explain what ROE means in one sentence.",
    temperature=0.3
)

print(f"Answer: {response['answer']}")

# Test structured output
intent = llm.classify_intent(
    user_query="Compare Apple and Microsoft's ROE",
    intent_classes=["comparison", "recommendation", "trend", "general"]
)

print(f"Intent: {intent}")
```

Run:
```bash
cd backend
python -m pytest tests/test_llm.py -v
```

### Step 2: Test RAG Service Standalone

Create `backend/tests/test_rag.py`:

```python
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService

# Initialize services
vector_store = VectorStore()
embedding_service = EmbeddingService()
llm_service = LLMService(provider="ollama", model="mistral")

# Create RAG service
rag = RAGService(
    vector_store=vector_store,
    embedding_service=embedding_service,
    llm_service=llm_service,
    top_k=5
)

# Test query
response = rag.query(
    user_query="What was Apple's revenue in 2023?",
    company_filter="AAPL"
)

print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence}")
print(f"Sources: {len(response.retrieved_chunks)} chunks")
print(f"Metrics: {response.metrics_used}")
```

### Step 3: Test via API

First, update `backend/app/main.py`:

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.analysis import router as analysis_router
from app.dependencies import startup_event, shutdown_event

app = FastAPI(
    title="Stock Recommendation API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lifecycle events
app.add_event_handler("startup", startup_event)
app.add_event_handler("shutdown", shutdown_event)

# Include routers
app.include_router(analysis_router)

@app.get("/")
def root():
    return {"message": "Stock Recommendation API", "status": "running"}
```

Start the server:
```bash
cd backend
uvicorn app.main:app --reload
```

Test endpoints:

**1. Simple Query:**
```bash
curl -X POST "http://localhost:8000/api/analysis/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was Apple'\''s revenue in 2023?",
    "company_ticker": "AAPL"
  }'
```

**2. Comparison:**
```bash
curl -X POST "http://localhost:8000/api/analysis/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "company_tickers": ["AAPL", "MSFT"],
    "metric_focus": "ROE",
    "year": 2023
  }'
```

**3. Recommendation:**
```bash
curl -X POST "http://localhost:8000/api/analysis/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "company_ticker": "AAPL",
    "investment_style": "growth"
  }'
```

---

## Example Responses

### Simple Query Response

```json
{
  "answer": "Based on Apple's 2023 annual report, total revenue was $394.3 billion, representing a 9% increase from the prior year's $365.8 billion. This growth was driven primarily by iPhone sales and services revenue.",
  "retrieved_chunks": [
    {
      "text": "Total net sales increased 9% to $394.3 billion...",
      "company_name": "Apple Inc.",
      "ticker": "AAPL",
      "year": 2023,
      "section_type": "financial_highlights",
      "similarity_score": 0.92,
      "page_numbers": [15, 16]
    }
  ],
  "metrics_used": {
    "sales": 394328000000,
    "sales_growth": 9.2
  },
  "confidence": 0.95,
  "reasoning": "High confidence due to direct citation from financial highlights section",
  "timestamp": "2026-02-07T10:30:00"
}
```

### Recommendation Response

```json
{
  "answer": "**RECOMMENDATION: BUY**\n\nApple demonstrates strong fundamentals for a growth investor:\n\n**Strengths:**\n- Revenue growth: 9.2% YoY\n- Net profit margin: 25.3% (exceptional)\n- ROE: 147% (outstanding capital efficiency)\n- EPS growth: 8.1%\n\n**Considerations:**\n- Premium valuation typical for quality growth\n- Some regulatory headwinds mentioned in risk factors\n\n**Verdict:** Apple meets growth investment criteria with strong revenue expansion, excellent profitability, and exceptional return on equity. The business shows pricing power and operational efficiency.",
  "confidence": 0.88,
  "reasoning": "Based on comprehensive analysis of 2023 financial data and comparison to growth investment criteria"
}
```

---

## RAG Query Types

### 1. **Factual Queries**

Questions about specific numbers or facts.

```python
response = rag.query(
    user_query="What was Tesla's net profit in 2023?",
    company_filter="TSLA"
)
```

**What happens:**
1. Retrieves financial highlights sections
2. Extracts net_profit metric
3. LLM formats answer with context

### 2. **Comparison Queries**

Compare multiple companies.

```python
response = rag.compare_companies(
    company_tickers=["AAPL", "MSFT", "GOOGL"],
    metric_focus="ROE"
)
```

**What happens:**
1. Retrieves data for each company
2. Extracts metrics for all
3. LLM generates side-by-side comparison

### 3. **Recommendation Queries**

Investment advice based on style.

```python
response = rag.get_recommendation(
    company_ticker="AAPL",
    investment_style="growth"
)
```

**What happens:**
1. Retrieves comprehensive financial data
2. Evaluates against style criteria
3. LLM generates BUY/HOLD/AVOID with reasoning

### 4. **Trend Queries**

Analysis over time (requires multi-year data).

```python
# Query 2022 and 2023 separately
response_2022 = rag.query("Apple revenue", company_filter="AAPL", year_filter=2022)
response_2023 = rag.query("Apple revenue", company_filter="AAPL", year_filter=2023)

# Then ask LLM to compare
```

---

## Customizing the RAG Flow

### Adjust Retrieval

Change number of chunks retrieved:

```python
rag = RAGService(
    vector_store=vector_store,
    embedding_service=embedding_service,
    llm_service=llm_service,
    top_k=10  # Retrieve more chunks for comprehensive analysis
)
```

### Adjust LLM Temperature

Control creativity vs consistency:

```python
# In rag_service.py, _generate_answer method
response = self.llm_service.generate(
    system_prompt=system_prompt,
    user_prompt=user_prompt,
    temperature=0.1,  # Very factual (0.0-0.3)
    # temperature=0.5,  # Balanced (default)
    # temperature=0.9,  # Creative (0.7-1.0)
    max_tokens=1000
)
```

### Custom System Prompts

Modify behavior by editing prompts in `rag_service.py`:

```python
def _build_system_prompt(self, intent: Dict[str, Any]) -> str:
    base_prompt = """You are a financial analyst AI with expertise in:
- Fundamental analysis
- Stock valuation
- Risk assessment

Your analysis style:
- Data-driven and objective
- Conservative estimates
- Highlight both opportunities and risks
- Always cite specific metrics

..."""
```

---

## Performance Optimization

### 1. Caching

Cache frequent queries:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_query(query: str, ticker: str) -> RAGResponse:
    return rag.query(query, company_filter=ticker)
```

### 2. Batch Processing

Process multiple queries in parallel:

```python
import asyncio

async def batch_analyze(queries: List[str]):
    tasks = [rag.query(q) for q in queries]
    return await asyncio.gather(*tasks)
```

### 3. Model Selection

Balance speed vs quality:

| Model | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| mistral | Fast | Good | Development, quick queries |
| llama3 | Medium | Better | Production, detailed analysis |
| gpt-4o-mini | Fast | Very Good | API option, cost-effective |
| claude-3.5-sonnet | Medium | Excellent | Complex analysis, best quality |

---

## Troubleshooting

### Issue 1: "No relevant chunks found"

**Cause:** Query doesn't match indexed data.

**Solution:**
```python
# Check what's indexed
stats = vector_store.get_stats()
print(stats['companies'])

# Make sure company data exists
years = vector_store.get_company_years("AAPL")
print(f"AAPL years available: {years}")
```

### Issue 2: "LLM connection failed"

**Cause:** Ollama not running.

**Solution:**
```bash
# Start Ollama
ollama serve

# Verify model exists
ollama list

# Pull model if missing
ollama pull mistral
```

### Issue 3: "Low confidence scores"

**Cause:** Poor retrieval quality.

**Solutions:**
- Ingest more data
- Improve chunk quality (better PDF processing)
- Increase `top_k` parameter
- Try different embedding model

---

## Next Steps

1. ✅ Implement RAG service
2. ✅ Add API routes
3. ⏭️ Create React frontend components
4. ⏭️ Add authentication
5. ⏭️ Deploy to production

See the frontend implementation guide for building the UI!

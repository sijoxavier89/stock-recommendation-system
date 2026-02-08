# Information Storage Strategy for Stock Recommendation System

## Overview
This document explains **what** information we store, **why** we store it, and **how** it's used in the RAG pipeline.

---

## Core Principle: The Metadata-Rich Chunk

Each document chunk stored in our vector database contains three essential components:

```python
{
    "text": "Apple's revenue grew by 9% year-over-year...",  # The actual content
    "embedding": [0.123, -0.456, ...],                     # 384-dim vector
    "metadata": {                                          # Rich context
        "company_name": "Apple Inc.",
        "ticker": "AAPL",
        "year": 2023,
        "section_type": "financial_highlights",
        "section_title": "Revenue Growth Analysis",
        "page_numbers": [15, 16],
        "chunk_index": 3,
        "file_path": "/data/annual_reports/aapl/2023.pdf",
        "metrics": [...]
    }
}
```

---

## Metadata Schema Explained

### 1. **Company Identification**

```python
"company_name": "Apple Inc."    # Full legal name
"ticker": "AAPL"                # Stock ticker symbol
```

**Why we store this:**
- Enables filtering by company during retrieval
- Allows multi-company comparisons
- Used in LLM prompts to provide company context
- Supports user queries like "Show me Apple's performance"

**How it's used:**
```python
# Example: Search only within Apple's documents
results = vector_store.search_by_company(
    query_embedding=query_emb,
    ticker="AAPL",
    n_results=5
)
```

---

### 2. **Temporal Information**

```python
"year": 2023                    # Fiscal year of the report
```

**Why we store this:**
- Critical for trend analysis ("revenue growth over 3 years")
- Enables time-based filtering
- Prevents mixing data from different years
- Supports comparative queries ("2022 vs 2023")

**How it's used:**
```python
# Example: Get only recent reports
results = vector_store.search(
    query_embedding=query_emb,
    where={"ticker": "AAPL", "year": {"$gte": 2020}}
)

# Example: LLM prompt includes temporal context
prompt = f"Based on {company_name}'s {year} annual report: {query}"
```

---

### 3. **Document Structure**

```python
"section_type": "financial_highlights"     # Semantic section category
"section_title": "Revenue Growth Analysis"  # Human-readable heading
"page_numbers": [15, 16]                   # Pages this chunk spans
"chunk_index": 3                           # Position within section
```

**Why we store this:**

#### Section Type
Annual reports have predictable structures. We categorize into:
- `executive_summary` - CEO letter, high-level overview
- `business_overview` - What the company does
- `financial_highlights` - Key metrics, performance summary
- `management_discussion` - MD&A section
- `risk_factors` - Risks and uncertainties
- `financial_statements` - Balance sheet, income statement

**Benefits:**
- Route queries to relevant sections (risks query → risk_factors section)
- Improve retrieval relevance
- Enable section-specific analysis

#### Page Numbers
**Benefits:**
- Citation/provenance ("found on pages 15-16")
- Allow users to verify information
- Debug retrieval quality

#### Chunk Index
**Benefits:**
- Understand context position within a section
- Reconstruct original section order if needed
- Track chunk boundaries

**How it's used:**
```python
# Example: Only search in risk sections
results = vector_store.search(
    query_embedding=query_emb,
    where={"section_type": "risk_factors"}
)

# Example: LLM prompt includes section context
retrieved_text = "Section: Risk Factors (pages 15-16)\n" + chunk_text
```

---

### 4. **Financial Metrics** (The Secret Weapon)

```python
"metrics": [
    {
        "name": "revenue",
        "value": 394000000000,    # $394 billion in base units
        "unit": "USD",
        "context": "...total revenue of $394 billion..."
    },
    {
        "name": "growth",
        "value": 9.5,
        "unit": "percentage"
    }
]
```

**Why we store this:**
This is what sets our system apart from basic RAG. We extract and structure financial data.

**Extracted metrics include:**
- Revenue, profit, EPS
- Growth rates (YoY)
- Margins (gross, operating, net)
- Key ratios (P/E, debt-to-equity)
- Employee count, market cap

**How it's used:**

1. **Quantitative Filtering**
```python
# Find high-growth companies
companies = get_companies_where(
    metrics_contain={"growth": {"$gte": 15}}  # >15% growth
)
```

2. **LLM Context Enhancement**
```python
# Provide structured data to LLM
context = f"""
Company: {company_name}
Year: {year}
Revenue: ${metrics['revenue']:,.0f}
Growth: {metrics['growth']}%

User question: {query}
"""
```

3. **Numerical Reasoning**
```python
# Calculate comparisons without LLM guessing
aapl_revenue_2023 = get_metric("AAPL", 2023, "revenue")
aapl_revenue_2022 = get_metric("AAPL", 2022, "revenue")
growth = ((aapl_revenue_2023 - aapl_revenue_2022) / aapl_revenue_2022) * 100
```

---

### 5. **Provenance**

```python
"file_path": "/data/annual_reports/aapl/2023.pdf"
```

**Why we store this:**
- Trace back to source document
- Support re-processing if needed
- Debugging and auditing
- Document versioning

---

## Chunking Strategy

### Why Chunking Matters

Annual reports are long (100+ pages). We can't embed the entire document as one vector because:
1. **Context window limits** - Embedding models have max length (~512 tokens for all-MiniLM-L6-v2)
2. **Semantic granularity** - Smaller chunks = more precise retrieval
3. **Computational efficiency** - Easier to search through focused chunks

### Our Approach

```python
chunk_size = 500 tokens        # ~375 words
chunk_overlap = 50 tokens      # ~38 words
```

**Why 500 tokens?**
- Large enough to capture complete thoughts
- Small enough for focused retrieval
- Fits within embedding model limits with buffer
- Balances precision vs context

**Why 50 token overlap?**
- Prevents splitting sentences across chunks
- Maintains context at chunk boundaries
- Example:

```
Chunk 1: "...the company achieved $394B in revenue. This represents..."
                                                    ↓ overlap region
Chunk 2: "...This represents a 9% increase over the prior year..."
```

### Semantic Chunking (Future Enhancement)

Current: Fixed-size chunks
Future: Semantic boundaries (paragraphs, sections)

```python
# Current
chunks = split_every_500_tokens(text)

# Future (better)
chunks = split_by_semantic_boundaries(
    text,
    target_size=500,
    boundaries=["paragraph", "section", "table"]
)
```

---

## Information We Intentionally DON'T Store

### 1. **Full Document Text**
**Don't store:** Entire PDF as one blob
**Do store:** Chunked, structured sections
**Reason:** Better retrieval granularity

### 2. **Images/Charts** (for now)
**Don't store:** Visual elements
**Future consideration:** Add multimodal embeddings
**Reason:** Complexity vs benefit trade-off

### 3. **Personal/Sensitive Data**
**Don't store:** SSNs, personal addresses
**Reason:** These shouldn't be in annual reports anyway

### 4. **Raw HTML/XML**
**Don't store:** PDF metadata, formatting tags
**Do store:** Clean extracted text
**Reason:** Focus on semantic content

---

## Retrieval Strategy: How Metadata Enhances Search

### Standard Vector Search (Basic RAG)
```
User: "What was Apple's revenue in 2023?"
    ↓
Embed query → Find similar chunks → Return top 5
```

**Problem:** Might return chunks from different companies/years

### Metadata-Enhanced Search (Our Approach)
```
User: "What was Apple's revenue in 2023?"
    ↓
Parse query → Extract filters (ticker=AAPL, year=2023)
    ↓
Embed query → Search WITH filters → Return top 5 from AAPL 2023 only
```

**Example:**
```python
# Automatically detected from query
filters = {
    "ticker": "AAPL",
    "year": 2023,
    "section_type": "financial_highlights"  # "revenue" keyword → financials
}

results = vector_store.search(
    query_embedding=embed("revenue performance"),
    where=filters,
    n_results=5
)
```

---

## Metadata for Multi-Company Analysis

### Comparative Queries

**Query:** "Compare Apple and Microsoft's revenue growth"

**Process:**
1. Search AAPL documents for revenue + growth
2. Search MSFT documents for revenue + growth
3. Extract metrics from both
4. Generate comparison

**Code:**
```python
aapl_results = search(query_emb, where={"ticker": "AAPL"})
msft_results = search(query_emb, where={"ticker": "MSFT"})

aapl_metrics = extract_metrics(aapl_results)
msft_metrics = extract_metrics(msft_results)

prompt = f"""
Compare these companies:

Apple ({aapl_metrics['year']}):
- Revenue: ${aapl_metrics['revenue']:,.0f}
- Growth: {aapl_metrics['growth']}%

Microsoft ({msft_metrics['year']}):
- Revenue: ${msft_metrics['revenue']:,.0f}
- Growth: {msft_metrics['growth']}%

Analysis: [LLM generates comparison]
"""
```

---

## Metadata for Temporal Analysis

### Trend Queries

**Query:** "How has Apple's revenue changed over the last 3 years?"

**Process:**
```python
years = [2021, 2022, 2023]
revenue_data = []

for year in years:
    results = search(
        query_emb,
        where={"ticker": "AAPL", "year": year, "section_type": "financial_highlights"}
    )
    revenue = extract_metric(results, "revenue")
    revenue_data.append({"year": year, "revenue": revenue})

# Generate trend analysis
prompt = f"""
Revenue trend for Apple:
{format_as_table(revenue_data)}

Analyze the trend and provide insights.
"""
```

---

## Storage Efficiency Considerations

### How Much Space?

**Per chunk:**
- Text: ~500 tokens = ~2KB
- Embedding: 384 floats × 4 bytes = 1.5KB
- Metadata: ~500 bytes (JSON)
- **Total: ~4KB per chunk**

**For 100 companies × 3 years × 100 pages:**
- Pages: 30,000
- Chunks: ~150,000 (5 chunks/page)
- Storage: 600MB (embeddings + text)
- ChromaDB overhead: ~200MB
- **Total: ~800MB**

**Conclusion:** Manageable on modern hardware

### Optimization Strategies

1. **Lazy loading** - Don't load all embeddings into memory
2. **Compression** - Use quantization for embeddings (4-bit)
3. **Pruning** - Remove old years if not needed
4. **Indexing** - ChromaDB uses HNSW for fast search

---

## Summary: The Metadata Decision Matrix

| Metadata Field | Storage Cost | Retrieval Value | Keep? |
|---|---|---|---|
| company_name | Low | High | ✅ Yes |
| ticker | Low | Critical | ✅ Yes |
| year | Low | Critical | ✅ Yes |
| section_type | Low | High | ✅ Yes |
| page_numbers | Low | Medium | ✅ Yes |
| chunk_index | Low | Low | ✅ Yes (cheap) |
| metrics | Medium | Very High | ✅ Yes |
| file_path | Low | Medium | ✅ Yes |
| full_text | High | Low | ❌ No (use chunks) |
| formatting | Low | Very Low | ❌ No |

**Guiding Principle:** Store metadata that enables better filtering, provides context to the LLM, or supports quantitative analysis. Avoid redundant or low-value fields.

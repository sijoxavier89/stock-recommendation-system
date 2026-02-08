# Quick Start: Implementing the Ingestion Pipeline

This guide walks you through implementing and testing the ingestion pipeline step-by-step.

---

## Step 1: Install Dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note:** First run will download the sentence-transformers model (~100MB). This happens automatically.

---

## Step 2: Set Up Directory Structure

```bash
mkdir -p data/annual_reports/aapl
mkdir -p data/chroma_db
```

---

## Step 3: Test PDF Processing (Standalone)

Create `test_pdf_processing.py`:

```python
from pathlib import Path
from pdf_processor import PDFProcessor

# Initialize processor
processor = PDFProcessor()

# Process a test PDF
result = processor.process_pdf(
    pdf_path=Path("data/annual_reports/aapl/2023.pdf"),
    company_name="Apple Inc.",
    ticker="AAPL",
    year=2023
)

# Inspect results
print(f"Sections found: {len(result['sections'])}")
print(f"Metrics extracted: {len(result['metrics'])}")
print("\nFirst section:")
print(result['sections'][0].section_type)
print(result['sections'][0].title)
print(result['sections'][0].content[:500])  # First 500 chars

# Test chunking
chunks = processor.prepare_for_embedding(result)
print(f"\nTotal chunks: {len(chunks)}")
print(f"Sample chunk metadata: {chunks[0]['metadata']}")
```

**Run:**
```bash
python test_pdf_processing.py
```

**Expected output:**
```
Sections found: 12
Metrics extracted: 47
First section: executive_summary
Total chunks: 234
```

---

## Step 4: Test Embedding Generation

Create `test_embeddings.py`:

```python
from embedding_service import EmbeddingService
import numpy as np

# Initialize service
embedding_service = EmbeddingService(
    model_name="all-MiniLM-L6-v2",
    device="cpu"
)

# Test single embedding
text = "Apple's revenue grew by 9% year-over-year to $394 billion."
embedding = embedding_service.embed_text(text)

print(f"Embedding shape: {embedding.shape}")
print(f"Embedding sample: {embedding[:10]}")

# Test batch embedding
texts = [
    "Revenue increased significantly.",
    "Profit margins remained stable.",
    "The company faces regulatory risks."
]

embeddings = embedding_service.embed_batch(texts)
print(f"\nBatch embeddings shape: {embeddings.shape}")

# Test similarity
query = "How did revenue perform?"
query_emb = embedding_service.embed_text(query)

similarities = [
    embedding_service.compute_similarity(query_emb, emb)
    for emb in embeddings
]

print(f"\nSimilarities to '{query}':")
for text, sim in zip(texts, similarities):
    print(f"  [{sim:.3f}] {text}")
```

**Run:**
```bash
python test_embeddings.py
```

**Expected output:**
```
Embedding shape: (384,)
Batch embeddings shape: (3, 384)
Similarities to 'How did revenue perform?':
  [0.856] Revenue increased significantly.
  [0.423] Profit margins remained stable.
  [0.312] The company faces regulatory risks.
```

---

## Step 5: Test Vector Store

Create `test_vector_store.py`:

```python
from vector_store import VectorStore
import numpy as np

# Initialize vector store
store = VectorStore(
    persist_directory="./data/chroma_db",
    collection_name="test_reports"
)

# Create test data
test_chunks = [
    {
        "text": "Apple reported revenue of $394 billion in fiscal 2023.",
        "embedding": np.random.rand(384),  # Replace with real embeddings
        "metadata": {
            "company_name": "Apple Inc.",
            "ticker": "AAPL",
            "year": 2023,
            "section_type": "financial_highlights",
            "section_title": "Revenue Overview",
            "page_numbers": [15, 16],
            "chunk_index": 0,
            "file_path": "/data/annual_reports/aapl/2023.pdf",
        }
    },
    {
        "text": "Microsoft's cloud revenue grew 28% to $111 billion.",
        "embedding": np.random.rand(384),
        "metadata": {
            "company_name": "Microsoft Corporation",
            "ticker": "MSFT",
            "year": 2023,
            "section_type": "financial_highlights",
            "section_title": "Cloud Revenue",
            "page_numbers": [22, 23],
            "chunk_index": 0,
            "file_path": "/data/annual_reports/msft/2023.pdf",
        }
    }
]

# Add to store
stats = store.add_documents(test_chunks)
print(f"Add stats: {stats}")

# Test search by company
query_emb = np.random.rand(384)
results = store.search_by_company(
    query_embedding=query_emb,
    ticker="AAPL",
    n_results=1
)

print(f"\nSearch results for AAPL:")
print(f"  Found: {len(results['documents'])} documents")
print(f"  Text: {results['documents'][0][:100]}...")

# Get all companies
companies = store.get_all_companies()
print(f"\nCompanies in store: {companies}")

# Clean up
store.reset_collection()
print("Collection reset")
```

**Run:**
```bash
python test_vector_store.py
```

---

## Step 6: Test Complete Pipeline

Create `test_full_pipeline.py`:

```python
from pathlib import Path
from ingestion_pipeline import IngestionPipeline
from pdf_processor import PDFProcessor
from embedding_service import EmbeddingService
from vector_store import VectorStore

# Initialize all services
vector_store = VectorStore(
    persist_directory="./data/chroma_db",
    collection_name="annual_reports"
)

embedding_service = EmbeddingService(
    model_name="all-MiniLM-L6-v2",
    device="cpu"
)

pdf_processor = PDFProcessor()

# Create pipeline
pipeline = IngestionPipeline(
    vector_store=vector_store,
    embedding_service=embedding_service,
    pdf_processor=pdf_processor
)

# Test single PDF ingestion
print("=" * 60)
print("Testing single PDF ingestion...")
print("=" * 60)

result = pipeline.ingest_pdf(
    pdf_path=Path("data/annual_reports/aapl/2023.pdf"),
    company_name="Apple Inc.",
    ticker="AAPL",
    year=2023,
    chunk_size=500,
    chunk_overlap=50,
    replace_existing=True  # Overwrite if exists
)

print(f"\nResult: {result['status']}")
if result['status'] == 'success':
    print(f"Sections extracted: {result['statistics']['sections_extracted']}")
    print(f"Chunks created: {result['statistics']['chunks_created']}")
    print(f"Chunks stored: {result['statistics']['chunks_stored']}")
    print(f"Duration: {result['duration_seconds']:.2f}s")

# Get overall stats
print("\n" + "=" * 60)
print("Vector Store Statistics")
print("=" * 60)

stats = pipeline.get_ingestion_stats()
print(f"Total chunks: {stats['total_chunks']}")
print(f"Total companies: {stats['total_companies']}")
print(f"\nCompanies:")
for company in stats['companies']:
    print(f"  {company['ticker']} ({company['company_name']}): {company['years']}")
```

**Run:**
```bash
python test_full_pipeline.py
```

**Expected output:**
```
============================================================
Testing single PDF ingestion...
============================================================
Step 1/4: Processing PDF...
  Extracted 12 sections
  Found 47 financial metrics
Step 2/4: Creating text chunks...
  Created 234 chunks
Step 3/4: Generating embeddings...
  [Progress bar...]
Step 4/4: Storing in vector database...
  Progress: 234/234
✓ Ingestion complete in 45.23s
  Processed 234 chunks at 5.2 chunks/sec

Result: success
Sections extracted: 12
Chunks created: 234
Chunks stored: 234
Duration: 45.23s

============================================================
Vector Store Statistics
============================================================
Total chunks: 234
Total companies: 1

Companies:
  AAPL (Apple Inc.): [2023]
```

---

## Step 7: Bulk Ingest (Optional)

If you have multiple PDFs organized in directories:

```python
# Bulk ingest from directory structure
results = pipeline.ingest_directory(
    base_dir=Path("data/annual_reports"),
    replace_existing=False
)

print(f"\nProcessed {len(results)} files")
for r in results:
    status_icon = "✓" if r['status'] == 'success' else "✗"
    print(f"  {status_icon} {r['ticker']} {r['year']}: {r['status']}")
```

---

## Step 8: Verify Data Quality

Create `verify_ingestion.py`:

```python
from vector_store import VectorStore
from embedding_service import EmbeddingService

store = VectorStore()
embedding_service = EmbeddingService()

# Test semantic search
queries = [
    "What was the company's revenue?",
    "What are the main risk factors?",
    "How did profit margins change?"
]

for query in queries:
    print(f"\nQuery: {query}")
    query_emb = embedding_service.embed_text(query)
    
    results = store.search(
        query_embedding=query_emb,
        n_results=3
    )
    
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'],
        results['metadatas'],
        results['distances']
    )):
        print(f"\n  Result {i+1} (similarity: {1-dist:.3f}):")
        print(f"    Section: {meta['section_type']}")
        print(f"    Pages: {meta['page_numbers']}")
        print(f"    Text: {doc[:150]}...")
```

**Run:**
```bash
python verify_ingestion.py
```

This will show you if the retrieval is finding semantically relevant chunks.

---

## Common Issues & Solutions

### Issue 1: "Model download failed"

**Solution:** Pre-download the model:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
```

### Issue 2: "PDF extraction returns empty text"

**Possible causes:**
- PDF is image-based (scanned document)
- PDF has security restrictions

**Solution:** Use OCR for image-based PDFs:
```bash
pip install pytesseract pdf2image
```

### Issue 3: "ChromaDB permission error"

**Solution:** Ensure write permissions:
```bash
chmod -R 755 data/chroma_db
```

### Issue 4: "Out of memory during embedding"

**Solution:** Reduce batch size:
```python
embedding_service.embed_batch(texts, batch_size=16)  # Instead of 32
```

---

## Next Steps

Once ingestion is working:

1. **Integrate with FastAPI** - Create API endpoints
2. **Add RAG service** - Connect retrieval to LLM
3. **Build frontend** - Upload interface
4. **Add monitoring** - Track ingestion progress
5. **Optimize** - Profile and improve performance

See `backend/app/routes/upload.py` for API integration example.

---

## Performance Benchmarks

Expected performance on standard hardware (CPU):

| Metric | Value |
|---|---|
| PDF Processing | ~30 pages/sec |
| Embedding Generation | ~100 chunks/sec |
| Vector Storage | ~500 chunks/sec |
| **Overall** | **~5-10 chunks/sec** |

For a 100-page annual report:
- ~500 chunks generated
- ~60 seconds total processing time

To improve:
- Use GPU for embeddings (10x faster)
- Parallel processing for multiple PDFs
- Cache embeddings for duplicate content

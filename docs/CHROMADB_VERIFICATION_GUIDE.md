# ChromaDB Verification Guide

Complete guide to check your ChromaDB installation and verify data is stored correctly.

---

## Quick Verification (30 seconds)

### Method 1: Simple Python Check

```python
# Save as: verify_chromadb.py
import chromadb
from pathlib import Path

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./data/chroma_db")

# Check collections
collections = client.list_collections()
print(f"Collections found: {len(collections)}")

for coll in collections:
    count = coll.count()
    print(f"  {coll.name}: {count} documents")
```

**Run:**
```bash
cd backend
python verify_chromadb.py
```

**Expected output if working:**
```
Collections found: 1
  annual_reports: 234 documents
```

**Expected output if empty:**
```
Collections found: 0
```

---

## Method 2: Use the Verification Scripts

I've created two verification scripts for you:

### Option A: Quick Check (Simple)

```bash
cd backend
python verify_chromadb.py
```

**What it checks:**
1. ✅ ChromaDB installed
2. ✅ Directory exists
3. ✅ Can connect
4. ✅ Collections exist
5. ✅ Has data

### Option B: Detailed Inspection (Comprehensive)

```bash
cd backend
python chromadb_inspector.py
```

**What it shows:**
- Installation status
- All collections
- Company breakdown by ticker
- Years available
- Section types
- Sample documents
- Metrics in documents

---

## Common Scenarios & Solutions

### Scenario 1: "No collections found"

**What it means:**
- ChromaDB is installed correctly
- But no data has been ingested yet

**Solution:**
Run the ingestion pipeline:

```bash
cd backend

# Option 1: Test with sample data
python -c "
from app.services.ingestion_pipeline import IngestionPipeline
from app.services.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService
from app.services.pdf_processor import PDFProcessor
from pathlib import Path

# Initialize services
vector_store = VectorStore()
embedding_service = EmbeddingService()
pdf_processor = PDFProcessor()

# Create pipeline
pipeline = IngestionPipeline(vector_store, embedding_service, pdf_processor)

# Ingest a PDF
result = pipeline.ingest_pdf(
    pdf_path=Path('data/annual_reports/aapl/2023.pdf'),
    company_name='Apple Inc.',
    ticker='AAPL',
    year=2023
)

print(result)
"

# Option 2: Bulk ingest from directory
python -m app.services.ingestion_pipeline
```

---

### Scenario 2: "Directory not found"

**What it means:**
- ChromaDB hasn't been initialized yet

**This is normal!** The directory is created automatically on first write.

**To create it:**
```bash
mkdir -p backend/data/chroma_db
```

Or just run ingestion - it will create the directory.

---

### Scenario 3: "ChromaDB not installed"

**Solution:**
```bash
pip install chromadb==0.4.18
```

---

### Scenario 4: "Connection failed"

**Possible causes:**
1. Permissions issue
2. Corrupted database
3. Wrong path

**Solutions:**

**Check permissions:**
```bash
ls -la backend/data/chroma_db
# Should show readable/writable
```

**Fix permissions:**
```bash
chmod -R 755 backend/data/chroma_db
```

**Try different path:**
```python
# In your code, use absolute path
import os
db_path = os.path.abspath("./data/chroma_db")
client = chromadb.PersistentClient(path=db_path)
```

---

## Manual Inspection Methods

### Method 1: Check Directory Contents

```bash
# Navigate to ChromaDB directory
cd backend/data/chroma_db

# List files
ls -lah

# Should see:
# - chroma.sqlite3 (main database)
# - Multiple .bin files (HNSW index files)
# - UUID-named directories (collection data)

# Check database size
du -sh .
```

**Healthy output:**
```
800K    ./chroma.sqlite3
2.3M    ./collection-uuid-123/
1.1M    ./collection-uuid-456/
```

---

### Method 2: Query ChromaDB Directly

```python
import chromadb

# Connect
client = chromadb.PersistentClient(path="./data/chroma_db")

# Get collection
collection = client.get_collection(name="annual_reports")

# Check count
print(f"Total documents: {collection.count()}")

# Get sample documents
results = collection.get(
    limit=5,
    include=["documents", "metadatas"]
)

# Display samples
for i, (doc, meta) in enumerate(zip(results["documents"], results["metadatas"]), 1):
    print(f"\n--- Document {i} ---")
    print(f"Company: {meta.get('ticker')}")
    print(f"Year: {meta.get('year')}")
    print(f"Text: {doc[:100]}...")
```

---

### Method 3: Check Specific Company

```python
import chromadb

client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_collection(name="annual_reports")

# Get all documents for Apple
results = collection.get(
    where={"ticker": "AAPL"},
    include=["metadatas"]
)

print(f"Apple chunks: {len(results['ids'])}")

# Get years available
years = set()
for meta in results["metadatas"]:
    if meta.get("year"):
        years.add(meta["year"])

print(f"Years available: {sorted(years)}")
```

---

### Method 4: Verify Embeddings

```python
import chromadb
import numpy as np

client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_collection(name="annual_reports")

# Get sample with embeddings
results = collection.get(
    limit=1,
    include=["embeddings", "metadatas"]
)

if results["embeddings"]:
    emb = results["embeddings"][0]
    print(f"Embedding dimension: {len(emb)}")
    print(f"Sample values: {emb[:10]}")
    print(f"Embedding norm: {np.linalg.norm(emb):.3f}")
    
    # Should be:
    # Dimension: 384 (for all-MiniLM-L6-v2)
    # Norm: ~1.0 (L2 normalized)
else:
    print("❌ No embeddings found!")
```

---

## Comprehensive Verification Checklist

Run this complete check:

```python
"""
Complete ChromaDB Health Check
"""
import chromadb
from pathlib import Path
import json

def full_health_check():
    print("CHROMADB HEALTH CHECK")
    print("=" * 60)
    
    db_path = Path("./data/chroma_db")
    
    # 1. Installation
    try:
        import chromadb
        print("✅ ChromaDB installed:", chromadb.__version__)
    except:
        print("❌ ChromaDB not installed")
        return
    
    # 2. Directory
    if db_path.exists():
        print(f"✅ Directory exists: {db_path}")
    else:
        print(f"❌ Directory missing: {db_path}")
        return
    
    # 3. Connection
    try:
        client = chromadb.PersistentClient(path=str(db_path))
        print("✅ Connected to ChromaDB")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return
    
    # 4. Collections
    collections = client.list_collections()
    if not collections:
        print("⚠️  No collections (run ingestion)")
        return
    
    print(f"✅ Collections: {len(collections)}")
    
    # 5. Data
    for coll in collections:
        count = coll.count()
        print(f"\n📊 Collection: {coll.name}")
        print(f"   Documents: {count}")
        
        if count == 0:
            continue
        
        # Sample
        sample = coll.get(limit=1, include=["metadatas", "embeddings"])
        
        # Metadata check
        if sample["metadatas"]:
            meta = sample["metadatas"][0]
            print(f"   ✅ Metadata present")
            print(f"      Company: {meta.get('ticker')}")
            print(f"      Year: {meta.get('year')}")
            
            # Metrics check
            metrics = meta.get("metrics")
            if metrics:
                print(f"   ✅ Metrics embedded")
                if isinstance(metrics, str):
                    try:
                        metrics = json.loads(metrics)
                        print(f"      Sample metrics: {list(metrics.keys())[:3]}")
                    except:
                        pass
        
        # Embeddings check
        if sample.get("embeddings"):
            emb = sample["embeddings"][0]
            print(f"   ✅ Embeddings present ({len(emb)} dims)")
        else:
            print(f"   ⚠️  No embeddings")
    
    print("\n" + "=" * 60)
    print("✅ HEALTH CHECK COMPLETE")

if __name__ == "__main__":
    full_health_check()
```

Save as `health_check.py` and run:
```bash
python health_check.py
```

---

## Expected File Structure

After successful ingestion, you should see:

```
backend/data/chroma_db/
├── chroma.sqlite3                    # Main database (metadata)
├── [uuid-1]/                         # Collection directory
│   ├── data_level0.bin              # HNSW index
│   ├── header.bin
│   └── link_lists.bin
└── [uuid-2]/                         # Another collection
    └── ...
```

**File sizes (approximate):**
- 100 documents: ~50 MB
- 500 documents: ~200 MB
- 1000+ documents: ~500 MB

---

## Troubleshooting Commands

### Reset ChromaDB (Start Fresh)

**⚠️ WARNING: This deletes all data!**

```bash
# Backup first (optional)
cp -r backend/data/chroma_db backend/data/chroma_db.backup

# Delete
rm -rf backend/data/chroma_db

# Reingest
python -m app.services.ingestion_pipeline
```

### Check for Corruption

```python
import sqlite3

db_path = "./data/chroma_db/chroma.sqlite3"
conn = sqlite3.connect(db_path)

# Check integrity
cursor = conn.execute("PRAGMA integrity_check")
result = cursor.fetchone()

if result[0] == "ok":
    print("✅ Database integrity OK")
else:
    print(f"❌ Database corrupted: {result}")

conn.close()
```

---

## Quick Reference: Common Commands

```bash
# Quick check
python verify_chromadb.py

# Detailed inspection
python chromadb_inspector.py

# Check specific company
python chromadb_inspector.py --ticker AAPL

# View samples
python chromadb_inspector.py --samples 5

# Check directory size
du -sh backend/data/chroma_db

# Count files
find backend/data/chroma_db -type f | wc -l

# View database with SQLite
sqlite3 backend/data/chroma_db/chroma.sqlite3 "SELECT * FROM collections;"
```

---

## Summary

**To verify ChromaDB is working:**

1. ✅ Run `python verify_chromadb.py`
2. ✅ Should see collection with document count
3. ✅ If empty, run ingestion pipeline
4. ✅ Verify specific company with inspector

**Red flags:**
- ❌ No `chroma.sqlite3` file → Database not initialized
- ❌ Count = 0 → No data ingested
- ❌ No embeddings → Ingestion incomplete
- ❌ Connection errors → Permission or path issues

**All good:**
- ✅ Collection exists
- ✅ Document count > 0
- ✅ Embeddings present
- ✅ Metadata has company/year/metrics

"""
Quick ChromaDB Verification Script
Simple script to check if ChromaDB is working and has data.
"""
import chromadb
from pathlib import Path


def quick_check():
    """Quick verification of ChromaDB installation and data."""
    
    print("\n" + "=" * 60)
    print("CHROMADB QUICK VERIFICATION")
    print("=" * 60 + "\n")
    
    # 1. Check installation
    print("1️⃣  Checking ChromaDB installation...")
    try:
        import chromadb
        print(f"   ✅ ChromaDB version: {chromadb.__version__}")
    except ImportError:
        print("   ❌ ChromaDB not installed")
        print("   Install with: pip install chromadb")
        return
    
    # 2. Check directory
    print("\n2️⃣  Checking data directory...")
    db_path = Path("./data/chroma_db")
    
    if db_path.exists():
        files = list(db_path.rglob("*"))
        size_mb = sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
        print(f"   ✅ Directory exists: {db_path}")
        print(f"   📁 Files: {len(files)}")
        print(f"   💾 Size: {size_mb:.2f} MB")
    else:
        print(f"   ⚠️  Directory not found: {db_path}")
        print(f"   Will be created on first use")
    
    # 3. Connect to ChromaDB
    print("\n3️⃣  Connecting to ChromaDB...")
    try:
        client = chromadb.PersistentClient(path=str(db_path))
        print(f"   ✅ Connected successfully")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return
    
    # 4. List collections
    print("\n4️⃣  Checking collections...")
    try:
        collections = client.list_collections()
        
        if not collections:
            print("   ⚠️  No collections found")
            print("\n   Next step: Run ingestion pipeline to add data")
            print("   Command: python backend/app/services/ingestion_pipeline.py")
            return
        
        print(f"   ✅ Found {len(collections)} collection(s)")
        
        for coll in collections:
            count = coll.count()
            print(f"\n   📊 Collection: {coll.name}")
            print(f"      Documents: {count}")
            
            if count > 0:
                # Get sample
                sample = coll.get(limit=1, include=["metadatas"])
                if sample["metadatas"]:
                    meta = sample["metadatas"][0]
                    print(f"      Sample company: {meta.get('ticker', 'N/A')}")
                    print(f"      Sample year: {meta.get('year', 'N/A')}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # 5. Summary
    print("\n" + "=" * 60)
    if collections and any(c.count() > 0 for c in collections):
        print("✅ SUCCESS: ChromaDB is working and has data!")
        print(f"\nTotal documents: {sum(c.count() for c in collections)}")
    else:
        print("⚠️  ChromaDB is installed but has no data")
        print("\nRun ingestion pipeline to add data:")
        print("  cd backend")
        print("  python -m app.services.ingestion_pipeline")
    print("=" * 60 + "\n")


def detailed_check():
    """Detailed inspection with company breakdown."""
    
    print("\n" + "=" * 60)
    print("CHROMADB DETAILED INSPECTION")
    print("=" * 60 + "\n")
    
    db_path = Path("./data/chroma_db")
    
    try:
        client = chromadb.PersistentClient(path=str(db_path))
        collections = client.list_collections()
        
        if not collections:
            print("❌ No collections found")
            return
        
        for coll in collections:
            print(f"\n📊 Collection: {coll.name}")
            print("-" * 60)
            
            count = coll.count()
            print(f"Total documents: {count}")
            
            if count == 0:
                print("(Empty collection)")
                continue
            
            # Get all metadata
            results = coll.get(
                limit=min(1000, count),
                include=["metadatas"]
            )
            
            # Analyze companies
            companies = {}
            for meta in results["metadatas"]:
                ticker = meta.get("ticker", "UNKNOWN")
                if ticker not in companies:
                    companies[ticker] = {
                        "name": meta.get("company_name", "Unknown"),
                        "years": set(),
                        "chunks": 0
                    }
                companies[ticker]["chunks"] += 1
                if meta.get("year"):
                    companies[ticker]["years"].add(meta["year"])
            
            # Display
            print(f"\nCompanies indexed: {len(companies)}")
            for ticker in sorted(companies.keys()):
                data = companies[ticker]
                years = sorted(list(data["years"]))
                print(f"  • {ticker} ({data['name']})")
                print(f"    Years: {years}")
                print(f"    Chunks: {data['chunks']}")
        
        print("\n" + "=" * 60 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    if "--detailed" in sys.argv or "-d" in sys.argv:
        detailed_check()
    else:
        quick_check()
        
        # Offer detailed check
        print("For detailed breakdown, run:")
        print("  python chromadb_inspector.py --detailed")

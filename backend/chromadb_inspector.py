"""
ChromaDB Verification and Inspection Tool
Check installation, view stored data, and verify ingestion.
"""
import chromadb
from chromadb.config import Settings
from pathlib import Path
import json
from typing import Dict, List, Any
from tabulate import tabulate
import sys


class ChromaDBInspector:
    """
    Utility to inspect and verify ChromaDB installation and data.
    """
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        Initialize inspector with ChromaDB path.
        
        Args:
            persist_directory: Path to ChromaDB storage
        """
        self.persist_directory = Path(persist_directory)
        self.client = None
        self.collection = None
    
    def check_installation(self) -> Dict[str, Any]:
        """
        Check if ChromaDB is properly installed and accessible.
        
        Returns:
            Dict with installation status
        """
        print("=" * 60)
        print("CHROMADB INSTALLATION CHECK")
        print("=" * 60)
        
        results = {
            "chromadb_installed": False,
            "version": None,
            "directory_exists": False,
            "directory_path": str(self.persist_directory),
            "client_initialized": False,
            "error": None
        }
        
        # Check if chromadb module is installed
        try:
            import chromadb
            results["chromadb_installed"] = True
            results["version"] = chromadb.__version__
            print(f"✅ ChromaDB installed: v{chromadb.__version__}")
        except ImportError as e:
            results["error"] = str(e)
            print(f"❌ ChromaDB not installed: {e}")
            return results
        
        # Check if directory exists
        if self.persist_directory.exists():
            results["directory_exists"] = True
            print(f"✅ Directory exists: {self.persist_directory}")
            
            # List files in directory
            files = list(self.persist_directory.rglob("*"))
            print(f"   Files found: {len(files)}")
            
            if files:
                print(f"   Directory size: {self._get_directory_size():.2f} MB")
        else:
            print(f"⚠️  Directory does not exist: {self.persist_directory}")
            print(f"   Will be created on first write")
        
        # Try to initialize client
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False)
            )
            results["client_initialized"] = True
            print(f"✅ ChromaDB client initialized successfully")
        except Exception as e:
            results["error"] = str(e)
            print(f"❌ Failed to initialize client: {e}")
            return results
        
        print()
        return results
    
    def list_collections(self) -> List[str]:
        """
        List all collections in ChromaDB.
        
        Returns:
            List of collection names
        """
        print("=" * 60)
        print("COLLECTIONS")
        print("=" * 60)
        
        if not self.client:
            print("❌ Client not initialized. Run check_installation() first.")
            return []
        
        try:
            collections = self.client.list_collections()
            
            if not collections:
                print("⚠️  No collections found")
                print("   Run ingestion pipeline to create collections")
                return []
            
            print(f"Found {len(collections)} collection(s):\n")
            
            collection_data = []
            for coll in collections:
                count = coll.count()
                metadata = coll.metadata or {}
                
                collection_data.append([
                    coll.name,
                    count,
                    metadata.get("hnsw:space", "N/A")
                ])
            
            print(tabulate(
                collection_data,
                headers=["Collection Name", "Documents", "Distance Metric"],
                tablefmt="grid"
            ))
            print()
            
            return [c.name for c in collections]
            
        except Exception as e:
            print(f"❌ Error listing collections: {e}")
            return []
    
    def inspect_collection(self, collection_name: str = "annual_reports") -> Dict[str, Any]:
        """
        Inspect a specific collection in detail.
        
        Args:
            collection_name: Name of collection to inspect
        
        Returns:
            Dict with collection statistics
        """
        print("=" * 60)
        print(f"COLLECTION INSPECTION: {collection_name}")
        print("=" * 60)
        
        if not self.client:
            print("❌ Client not initialized. Run check_installation() first.")
            return {}
        
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception as e:
            print(f"❌ Collection '{collection_name}' not found: {e}")
            print("   Available collections:", self.list_collections())
            return {}
        
        # Get collection stats
        stats = {
            "name": collection_name,
            "total_documents": 0,
            "companies": [],
            "years": set(),
            "sections": {},
            "sample_metadata": None
        }
        
        try:
            stats["total_documents"] = self.collection.count()
            print(f"📊 Total documents: {stats['total_documents']}")
            
            if stats["total_documents"] == 0:
                print("⚠️  Collection is empty")
                print("   Run ingestion pipeline to add data")
                return stats
            
            # Get all documents (or sample if too many)
            limit = min(1000, stats["total_documents"])
            results = self.collection.get(
                limit=limit,
                include=["metadatas"]
            )
            
            # Analyze metadata
            companies = {}
            sections_count = {}
            
            for metadata in results["metadatas"]:
                # Track companies
                ticker = metadata.get("ticker", "UNKNOWN")
                company_name = metadata.get("company_name", "Unknown")
                year = metadata.get("year")
                
                if ticker not in companies:
                    companies[ticker] = {
                        "name": company_name,
                        "years": set(),
                        "chunks": 0
                    }
                
                companies[ticker]["chunks"] += 1
                if year:
                    companies[ticker]["years"].add(year)
                    stats["years"].add(year)
                
                # Track sections
                section = metadata.get("section_type", "unknown")
                sections_count[section] = sections_count.get(section, 0) + 1
            
            # Convert sets to sorted lists
            stats["companies"] = [
                {
                    "ticker": ticker,
                    "name": data["name"],
                    "years": sorted(list(data["years"])),
                    "chunks": data["chunks"]
                }
                for ticker, data in sorted(companies.items())
            ]
            
            stats["years"] = sorted(list(stats["years"]))
            stats["sections"] = sections_count
            stats["sample_metadata"] = results["metadatas"][0] if results["metadatas"] else None
            
            # Display results
            print(f"\n📈 Companies: {len(companies)}")
            print(f"📅 Years: {', '.join(map(str, stats['years']))}")
            print()
            
            # Company breakdown
            print("COMPANIES:")
            company_table = [
                [
                    c["ticker"],
                    c["name"][:30],
                    ", ".join(map(str, c["years"])),
                    c["chunks"]
                ]
                for c in stats["companies"]
            ]
            
            print(tabulate(
                company_table,
                headers=["Ticker", "Company Name", "Years", "Chunks"],
                tablefmt="grid"
            ))
            print()
            
            # Section breakdown
            print("SECTIONS:")
            section_table = [
                [section, count]
                for section, count in sorted(sections_count.items(), key=lambda x: x[1], reverse=True)
            ]
            
            print(tabulate(
                section_table,
                headers=["Section Type", "Chunks"],
                tablefmt="grid"
            ))
            print()
            
            return stats
            
        except Exception as e:
            print(f"❌ Error inspecting collection: {e}")
            return stats
    
    def view_sample_documents(
        self,
        collection_name: str = "annual_reports",
        ticker: str = None,
        n_samples: int = 3
    ):
        """
        View sample documents from collection.
        
        Args:
            collection_name: Collection to sample from
            ticker: Optional ticker to filter by
            n_samples: Number of samples to show
        """
        print("=" * 60)
        print(f"SAMPLE DOCUMENTS")
        print("=" * 60)
        
        if not self.client:
            print("❌ Client not initialized")
            return
        
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # Build filter
            where = {"ticker": ticker.upper()} if ticker else None
            
            # Get samples
            results = collection.get(
                limit=n_samples,
                where=where,
                include=["documents", "metadatas", "embeddings"]
            )
            
            if not results["documents"]:
                print("⚠️  No documents found")
                return
            
            for i, (doc, meta, emb) in enumerate(zip(
                results["documents"],
                results["metadatas"],
                results.get("embeddings", [None] * len(results["documents"]))
            ), 1):
                print(f"\n📄 SAMPLE {i}/{n_samples}")
                print("-" * 60)
                
                # Metadata
                print(f"Company: {meta.get('company_name', 'N/A')} ({meta.get('ticker', 'N/A')})")
                print(f"Year: {meta.get('year', 'N/A')}")
                print(f"Section: {meta.get('section_type', 'N/A')}")
                print(f"Pages: {meta.get('page_numbers', 'N/A')}")
                
                # Document text
                print(f"\nText (first 300 chars):")
                print(doc[:300] + "..." if len(doc) > 300 else doc)
                
                # Embedding info
                if emb:
                    print(f"\nEmbedding: {len(emb)} dimensions")
                    print(f"Sample values: {emb[:5]}")
                
                # Metrics if available
                metrics = meta.get("metrics")
                if metrics:
                    if isinstance(metrics, str):
                        try:
                            metrics = json.loads(metrics)
                        except:
                            pass
                    
                    if isinstance(metrics, dict):
                        print(f"\nMetrics:")
                        for key, value in list(metrics.items())[:5]:
                            print(f"  {key}: {value}")
            
            print("\n" + "=" * 60)
            
        except Exception as e:
            print(f"❌ Error viewing samples: {e}")
    
    def search_test(
        self,
        collection_name: str = "annual_reports",
        query: str = "revenue growth",
        n_results: int = 3
    ):
        """
        Test search functionality.
        
        Args:
            collection_name: Collection to search
            query: Search query
            n_results: Number of results
        """
        print("=" * 60)
        print(f"SEARCH TEST")
        print("=" * 60)
        print(f"Query: '{query}'")
        print()
        
        if not self.client:
            print("❌ Client not initialized")
            return
        
        try:
            collection = self.client.get_collection(name=collection_name)
            
            # Note: ChromaDB requires embeddings for search
            # This is a simple test - in production, use embedding_service
            print("⚠️  Note: This requires embeddings to work properly")
            print("   Use the embedding_service for production searches")
            print()
            
            # Try to search (this will fail if no default embedding function)
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                
                print(f"Found {len(results['documents'][0])} results:\n")
                
                for i, (doc, meta, dist) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                ), 1):
                    print(f"Result {i}:")
                    print(f"  Similarity: {1 - dist:.3f}")
                    print(f"  Company: {meta.get('ticker', 'N/A')}")
                    print(f"  Section: {meta.get('section_type', 'N/A')}")
                    print(f"  Text: {doc[:150]}...")
                    print()
                
            except Exception as e:
                print(f"❌ Search failed: {e}")
                print("   ChromaDB collection has no default embedding function")
                print("   Use embedding_service.embed_text() + vector_store.search()")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def _get_directory_size(self) -> float:
        """Get total size of ChromaDB directory in MB."""
        total = sum(
            f.stat().st_size
            for f in self.persist_directory.rglob("*")
            if f.is_file()
        )
        return total / (1024 * 1024)
    
    def full_report(self, collection_name: str = "annual_reports"):
        """
        Generate complete inspection report.
        
        Args:
            collection_name: Collection to inspect
        """
        self.check_installation()
        self.list_collections()
        stats = self.inspect_collection(collection_name)
        
        if stats.get("total_documents", 0) > 0:
            self.view_sample_documents(collection_name, n_samples=2)
        
        print("\n" + "=" * 60)
        print("VERIFICATION COMPLETE")
        print("=" * 60)
        
        if stats.get("total_documents", 0) == 0:
            print("\n⚠️  NO DATA FOUND")
            print("\nNext steps:")
            print("1. Run the ingestion pipeline:")
            print("   python -m app.services.ingestion_pipeline")
            print("\n2. Or test with sample data:")
            print("   python test_chromadb_verification.py --add-sample")
        else:
            print(f"\n✅ ChromaDB is working with {stats['total_documents']} documents")
            print(f"   Companies: {len(stats['companies'])}")
            print(f"   Years: {stats['years']}")


def main():
    """CLI interface for ChromaDB inspection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect ChromaDB installation and data")
    parser.add_argument(
        "--path",
        default="./data/chroma_db",
        help="Path to ChromaDB directory"
    )
    parser.add_argument(
        "--collection",
        default="annual_reports",
        help="Collection name to inspect"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of sample documents to view"
    )
    parser.add_argument(
        "--ticker",
        help="Filter samples by ticker symbol"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick check (skip samples)"
    )
    
    args = parser.parse_args()
    
    # Create inspector
    inspector = ChromaDBInspector(persist_directory=args.path)
    
    # Run checks
    if args.quick:
        inspector.check_installation()
        inspector.list_collections()
        inspector.inspect_collection(args.collection)
    else:
        inspector.full_report(args.collection)
        
        if args.ticker:
            print(f"\n📌 Filtered view for {args.ticker}:")
            inspector.view_sample_documents(
                args.collection,
                ticker=args.ticker,
                n_samples=args.samples
            )


if __name__ == "__main__":
    # If run directly, do full report
    if len(sys.argv) == 1:
        inspector = ChromaDBInspector()
        inspector.full_report()
    else:
        main()

"""
Standalone Ingestion Test Script
Run this directly to test the ingestion pipeline without import issues.

Usage:
    python test_ingestion.py
"""
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Now import services
from app.services.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService
from app.services.pdf_processor import PDFProcessor
from app.services.ingestion_pipeline import IngestionPipeline


def test_single_pdf():
    """Test ingesting a single PDF."""
    print("=" * 60)
    print("TESTING SINGLE PDF INGESTION")
    print("=" * 60)
    
    # Initialize services
    print("\n1. Initializing services...")
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
    
    print("   ✓ Services initialized")
    
    # Check if we have a test PDF
    test_pdf = Path("./data/annual_reports/MSFT/2023_2d6c6afe_2024_Annual_Report.pdf")
    
    if not test_pdf.exists():
        print(f"\n❌ Test PDF not found: {test_pdf}")
        print("\nPlease add a PDF to test:")
        print(f"  mkdir -p {test_pdf.parent}")
        print(f"  # Place a PDF at {test_pdf}")
        return
    
    print(f"\n2. Found test PDF: {test_pdf}")
    
    # Ingest the PDF
    print("\n3. Starting ingestion...")
    result = pipeline.ingest_pdf(
        pdf_path=test_pdf,
        company_name="Apple Inc.",
        ticker="AAPL",
        year=2023,
        chunk_size=500,
        chunk_overlap=50,
        replace_existing=True  # Replace if already exists
    )
    
    print("\n" + "=" * 60)
    print("INGESTION RESULT")
    print("=" * 60)
    
    if result["status"] == "success":
        print("✅ SUCCESS!")
        print(f"\nStatistics:")
        print(f"  Sections extracted: {result['statistics']['sections_extracted']}")
        print(f"  Metrics found: {result['statistics']['metrics_found']}")
        print(f"  Chunks created: {result['statistics']['chunks_created']}")
        print(f"  Chunks stored: {result['statistics']['chunks_stored']}")
        print(f"  Total pages: {result['statistics']['total_pages']}")
        print(f"  Duration: {result['duration_seconds']:.2f} seconds")
        print(f"  Speed: {result['chunks_per_second']:.1f} chunks/sec")
    else:
        print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
    
    # Show vector store stats
    print("\n" + "=" * 60)
    print("VECTOR STORE STATISTICS")
    print("=" * 60)
    
    stats = pipeline.get_ingestion_stats()
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total companies: {stats['total_companies']}")
    print(f"\nCompanies indexed:")
    for company in stats['companies']:
        print(f"  • {company['ticker']} ({company['company_name']})")
        print(f"    Years: {company['years']}")
        print(f"    Chunks: {len([c for c in stats['companies'] if c['ticker'] == company['ticker']])}")


def test_directory_ingestion():
    """Test ingesting all PDFs from a directory."""
    print("=" * 60)
    print("TESTING DIRECTORY INGESTION")
    print("=" * 60)
    
    # Initialize services
    print("\n1. Initializing services...")
    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    pdf_processor = PDFProcessor()
    
    pipeline = IngestionPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        pdf_processor=pdf_processor
    )
    
    print("   ✓ Services initialized")
    
    # Check directory
    reports_dir = Path("./data/annual_reports")
    
    if not reports_dir.exists():
        print(f"\n❌ Directory not found: {reports_dir}")
        print("\nCreate directory structure:")
        print(f"  mkdir -p {reports_dir}/aapl")
        print(f"  mkdir -p {reports_dir}/msft")
        print(f"  # Add PDFs named by year (e.g., 2023.pdf)")
        return
    
    # Count PDFs
    pdf_count = len(list(reports_dir.rglob("*.pdf")))
    
    if pdf_count == 0:
        print(f"\n⚠️  No PDFs found in {reports_dir}")
        return
    
    print(f"\n2. Found {pdf_count} PDF(s) in {reports_dir}")
    
    # Ingest
    print("\n3. Starting bulk ingestion...")
    results = pipeline.ingest_directory(
        base_dir=reports_dir,
        replace_existing=False
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("BULK INGESTION COMPLETE")
    print("=" * 60)
    
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    
    print(f"\nResults:")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"  ⊝ Skipped: {skipped}")
    
    if successful > 0:
        total_time = sum(r.get("duration_seconds", 0) for r in results if r["status"] == "success")
        total_chunks = sum(r.get("statistics", {}).get("chunks_created", 0) for r in results if r["status"] == "success")
        print(f"\n  Total time: {total_time:.2f}s")
        print(f"  Total chunks: {total_chunks}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ingestion pipeline")
    parser.add_argument(
        "--mode",
        choices=["single", "directory"],
        default="single",
        help="Test mode: single PDF or directory"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "single":
            test_single_pdf()
        else:
            test_directory_ingestion()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
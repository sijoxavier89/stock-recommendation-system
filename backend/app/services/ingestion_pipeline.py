"""
Complete Ingestion Pipeline
Orchestrates the entire flow: PDF → Text → Embeddings → Vector Store
"""
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import sys

# Handle both relative imports (when run as module) and absolute imports (when run directly)
try:
    from .pdf_processor import PDFProcessor
    from .embedding_service import EmbeddingService
    from .vector_store import VectorStore
except ImportError:
    # If relative imports fail, try absolute imports
    import os
    # Add parent directory to path
    current_dir = Path(__file__).parent
    backend_dir = current_dir.parent.parent
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))
    
    from app.services.pdf_processor import PDFProcessor
    from app.services.embedding_service import EmbeddingService
    from app.services.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Complete pipeline for ingesting annual reports into the vector store.
    
    Architecture:
    1. PDF Processing: Extract text, detect sections, extract metrics
    2. Chunking: Split into semantic chunks with metadata
    3. Embedding: Generate vector embeddings for each chunk
    4. Storage: Persist in ChromaDB with full metadata
    
    What Information We Store:
    -------------------------
    For each chunk of text, we store:
    
    1. The Text Itself
       - Raw text content from the PDF
       - Pre-chunked with overlap for context preservation
    
    2. The Vector Embedding
       - 384-dimensional vector (for all-MiniLM-L6-v2)
       - L2 normalized for cosine similarity
    
    3. Rich Metadata
       - company_name: Full company name
       - ticker: Stock ticker symbol
       - year: Fiscal year of the report
       - section_type: Category (financial_highlights, risks, etc.)
       - section_title: Human-readable section heading
       - page_numbers: List of pages this chunk spans
       - chunk_index: Position within the section
       - file_path: Original PDF location
       - metrics: Any financial metrics found in this chunk
    
    Why This Metadata Matters:
    --------------------------
    - Filtering: Users can filter by company, year, section
    - Context: LLM gets relevant metadata in prompts
    - Provenance: Track back to original document/page
    - Metrics: Enable quantitative analysis and comparisons
    - Deduplication: Detect and handle re-uploads
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        pdf_processor: PDFProcessor
    ):
        """
        Initialize the pipeline with all required services.
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.pdf_processor = pdf_processor
        
        logger.info("Ingestion pipeline initialized")
    
    def ingest_pdf(
        self,
        pdf_path: Path,
        ticker: str,
        year: int,
        company_name: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        replace_existing: bool = False
    ) -> Dict:
        """
        Complete ingestion pipeline for a single PDF.
        
        Args:
            pdf_path: Path to the PDF file
            ticker: Stock ticker (e.g., "AAPL")
            year: Fiscal year
            company_name: Full company name (e.g., "Apple Inc."). If None, will be inferred from ticker.
            chunk_size: Target size for text chunks (in tokens)
            chunk_overlap: Overlap between chunks (in tokens)
            replace_existing: If True, delete existing data for this company/year
        
        Returns:
            Dictionary with ingestion statistics and status
        """
        start_time = datetime.now()
        logger.info(f"Starting ingestion: {ticker} {year} - {pdf_path}")
        
        try:
            # Step 1: Check if this document already exists
            if replace_existing:
                deleted = self.vector_store.delete_company_year(ticker, year)
                logger.info(f"Deleted {deleted} existing chunks")
            else:
                existing_years = self.vector_store.get_company_years(ticker)
                if year in existing_years:
                    logger.warning(f"{ticker} {year} already exists. Use replace_existing=True to overwrite.")
                    return {
                        "status": "skipped",
                        "reason": "already_exists",
                        "ticker": ticker,
                        "year": year
                    }
            
            # Step 2: Process PDF - Extract text and structure
            logger.info("Step 1/4: Processing PDF...")
            if company_name is None:
                company_name = self._ticker_to_company_name(ticker)

            processed_data = self.pdf_processor.process_pdf(
                pdf_path=pdf_path,
                ticker=ticker,
                year=year,
                company_name=company_name
            )
            logger.info(f"  Extracted {len(processed_data['sections'])} sections")
            logger.info(f"  Found {len(processed_data['metrics'])} financial metrics")
            
            # Step 3: Prepare chunks with metadata
            logger.info("Step 2/4: Creating text chunks...")
            chunks = self.pdf_processor.prepare_for_embedding(
                processed_data=processed_data,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            logger.info(f"  Created {len(chunks)} chunks")
            
            # Step 4: Generate embeddings
            logger.info("Step 3/4: Generating embeddings...")
            chunks_with_embeddings = self.embedding_service.embed_documents(
                chunks=chunks,
                text_key="text",
                batch_size=32
            )
            
            # Step 5: Store in vector database
            logger.info("Step 4/4: Storing in vector database...")
            add_stats = self.vector_store.add_documents(
                chunks=chunks_with_embeddings,
                batch_size=100
            )
            
            # Calculate timing
            duration = (datetime.now() - start_time).total_seconds()
            
            result = {
                "status": "success",
                "ticker": ticker,
                "year": year,
                "company_name": company_name,
                "pdf_path": str(pdf_path),
                "statistics": {
                    "sections_extracted": len(processed_data['sections']),
                    "metrics_found": len(processed_data['metrics']),
                    "chunks_created": len(chunks),
                    "chunks_stored": add_stats['added'],
                    "chunks_failed": add_stats['failed'],
                    "total_pages": processed_data['metadata']['total_pages'],
                },
                "duration_seconds": duration,
                "chunks_per_second": len(chunks) / duration if duration > 0 else 0,
            }
            
            logger.info(f"✓ Ingestion complete in {duration:.2f}s")
            logger.info(f"  Processed {len(chunks)} chunks at {result['chunks_per_second']:.1f} chunks/sec")
            
            return result
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            return {
                "status": "failed",
                "ticker": ticker,
                "year": year,
                "error": str(e),
                "duration_seconds": (datetime.now() - start_time).total_seconds()
            }
    
    def ingest_directory(
        self,
        base_dir: Path,
        replace_existing: bool = False
    ) -> List[Dict]:
        """
        Ingest all PDFs from a directory structure.
        
        Expected structure:
        base_dir/
        ├── aapl/
        │   ├── 2022.pdf
        │   └── 2023.pdf
        ├── tsla/
        │   └── 2023.pdf
        
        The directory name is used as the ticker symbol.
        The filename (without .pdf) is used as the year.
        
        Returns:
            List of ingestion results for each file
        """
        logger.info(f"Scanning directory: {base_dir}")
        
        results = []
        
        # Scan for PDFs
        for company_dir in base_dir.iterdir():
            if not company_dir.is_dir():
                continue
            
            ticker = company_dir.name.upper()
            
            for pdf_file in company_dir.glob("*.pdf"):
                try:
                    # Extract year from filename
                    year = int(pdf_file.stem)
                    
                    # Infer company name from ticker (you might want a lookup table)
                    company_name = self._ticker_to_company_name(ticker)
                    
                    logger.info(f"\nProcessing: {ticker} {year}")
                    
                    result = self.ingest_pdf(
                        pdf_path=pdf_file,
                        company_name=company_name,
                        ticker=ticker,
                        year=year,
                        replace_existing=replace_existing
                    )
                    
                    results.append(result)
                    
                except ValueError:
                    logger.warning(f"Skipping {pdf_file}: filename must be a year (e.g., 2023.pdf)")
                    continue
                except Exception as e:
                    logger.error(f"Failed to process {pdf_file}: {e}")
                    results.append({
                        "status": "failed",
                        "ticker": ticker,
                        "file": str(pdf_file),
                        "error": str(e)
                    })
        
        # Summary
        successful = sum(1 for r in results if r["status"] == "success")
        failed = sum(1 for r in results if r["status"] == "failed")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Bulk ingestion complete:")
        logger.info(f"  ✓ Successful: {successful}")
        logger.info(f"  ✗ Failed: {failed}")
        logger.info(f"  ⊝ Skipped: {skipped}")
        logger.info(f"{'='*60}\n")
        
        return results
    
    def _ticker_to_company_name(self, ticker: str) -> str:
        """
        Convert ticker to company name.
        In production, use a proper lookup table or API.
        """
        # Simple mapping for common companies
        ticker_map = {
            "AAPL": "Apple Inc.",
            "MSFT": "Microsoft Corporation",
            "GOOGL": "Alphabet Inc.",
            "AMZN": "Amazon.com Inc.",
            "TSLA": "Tesla Inc.",
            "META": "Meta Platforms Inc.",
            "NVDA": "NVIDIA Corporation",
            "JPM": "JPMorgan Chase & Co.",
            "V": "Visa Inc.",
            "WMT": "Walmart Inc.",
        }
        
        return ticker_map.get(ticker, f"{ticker} Inc.")
    
    def get_ingestion_stats(self) -> Dict:
        """
        Get overall statistics about ingested data.
        """
        return self.vector_store.get_stats()


# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest annual reports into vector store")
    parser.add_argument(
        "--pdf",
        type=str,
        help="Path to a single PDF file to ingest"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Company ticker symbol (required with --pdf)"
    )
    parser.add_argument(
        "--company",
        type=str,
        help="Company name (required with --pdf)"
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Fiscal year (required with --pdf)"
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="./data/annual_reports",
        help="Directory containing PDFs (default: ./data/annual_reports)"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing data for the same company/year"
    )
    
    args = parser.parse_args()
    
    # Initialize services
    print("Initializing services...")
    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    pdf_processor = PDFProcessor()
    
    # Create pipeline
    pipeline = IngestionPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        pdf_processor=pdf_processor
    )
    
    # Single PDF or directory ingestion
    if args.pdf:
        # Single PDF mode
        if not all([args.ticker, args.company, args.year]):
            print("Error: --ticker, --company, and --year are required with --pdf")
            sys.exit(1)
        
        result = pipeline.ingest_pdf(
            pdf_path=Path(args.pdf),
            company_name=args.company,
            ticker=args.ticker,
            year=args.year,
            replace_existing=args.replace
        )
        
        print("\nIngestion Result:")
        print(result)
    else:
        # Directory mode
        results = pipeline.ingest_directory(
            base_dir=Path(args.directory),
            replace_existing=args.replace
        )
        
        print(f"\nProcessed {len(results)} files")
    
    # Show final stats
    print("\nVector Store Statistics:")
    stats = pipeline.get_ingestion_stats()
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Companies: {stats['total_companies']}")
    print(f"\nCompanies in store:")
    for company in stats['companies']:
        print(f"  {company['ticker']}: {company['years']}")
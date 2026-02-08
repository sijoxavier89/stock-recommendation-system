"""
Vector Store Service
Manages ChromaDB for storing and retrieving document embeddings.
"""
from typing import List, Dict, Optional, Any
import numpy as np
import logging
from pathlib import Path
import json
from ..core.config import CHROMA_DB_DIR

# Use CHROMA_DB_DIR as the persist directory for chromadb
PERSIST_DIRECTORY = CHROMA_DB_DIR

# Example chromadb client initialization (adjust to your existing code)
try:
    import chromadb
    from chromadb.config import Settings
    _CHROMA_CLIENT = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(PERSIST_DIRECTORY)))
except Exception:
    _CHROMA_CLIENT = None
    # existing code should handle missing client gracefully

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use CHROMA_DB_DIR as the persist directory for chromadb
PERSIST_DIRECTORY = CHROMA_DB_DIR


class VectorStore:
    """
    ChromaDB-based vector store for annual report embeddings.
    
    Key Design Decisions:
    1. Collections: One collection per document type (annual_reports)
    2. Metadata Schema: Company, year, section, page numbers, metrics
    3. Persistence: Local disk storage for durability
    4. Filtering: Support complex metadata filters
    """
    
    # Metadata schema - what information we store for each chunk
    METADATA_SCHEMA = {
        "company_name": "str",        # e.g., "Apple Inc."
        "ticker": "str",              # e.g., "AAPL"
        "year": "int",                # e.g., 2023
        "section_type": "str",        # e.g., "financial_highlights"
        "section_title": "str",       # e.g., "Revenue Growth Analysis"
        "page_numbers": "list[int]",  # e.g., [15, 16, 17]
        "chunk_index": "int",         # Position within section
        "file_path": "str",           # Original PDF location
        # Optional fields:
        "metrics": "list[dict]",      # Extracted financial metrics
    }
    
    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = "annual_reports"
    ):
        """
        Initialize ChromaDB client and collection.
        
        Args:
            persist_directory: Where to store the database
            collection_name: Name of the collection to use
        """
        # Use provided directory or fall back to the configured PERSIST_DIRECTORY
        self.persist_directory = Path(persist_directory or str(PERSIST_DIRECTORY))
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # Get or create collection
        self.collection_name = collection_name
        self.collection = self._get_or_create_collection()
        
        logger.info(f"Vector store initialized at {self.persist_directory}")
        logger.info(f"Collection '{collection_name}' has {self.collection.count()} items")
    
    def _get_or_create_collection(self):
        """
        Get existing collection or create a new one.
        Uses cosine similarity for distance metric.
        """
        try:
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None  # We'll provide embeddings manually
            )
            logger.info(f"Using existing collection: {self.collection_name}")
        except:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # Cosine similarity
                embedding_function=None
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def add_documents(
        self,
        chunks: List[Dict],
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of dicts with 'text', 'embedding', 'metadata'
            batch_size: Number of docs to add at once
        
        Returns:
            Statistics about the insertion
        """
        total_chunks = len(chunks)
        logger.info(f"Adding {total_chunks} chunks to vector store")
        
        added = 0
        failed = 0
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            
            try:
                # Prepare data for ChromaDB
                ids = [self._generate_id(chunk, i + j) for j, chunk in enumerate(batch)]
                embeddings = [chunk["embedding"].tolist() for chunk in batch]
                documents = [chunk["text"] for chunk in batch]
                metadatas = [self._prepare_metadata(chunk["metadata"]) for chunk in batch]
                
                # Add to collection
                self.collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                
                added += len(batch)
                logger.info(f"Progress: {added}/{total_chunks}")
                
            except Exception as e:
                logger.error(f"Failed to add batch {i // batch_size}: {e}")
                failed += len(batch)
        
        return {
            "total": total_chunks,
            "added": added,
            "failed": failed,
            "collection_size": self.collection.count()
        }
    
    def _generate_id(self, chunk: Dict, index: int) -> str:
        """
        Generate unique ID for a chunk.
        Format: {ticker}_{year}_{section}_{chunk_index}
        """
        metadata = chunk["metadata"]
        return (
            f"{metadata['ticker']}_"
            f"{metadata['year']}_"
            f"{metadata['section_type']}_"
            f"{metadata.get('chunk_index', index)}"
        )
    
    def _prepare_metadata(self, metadata: Dict) -> Dict:
        """
        Prepare metadata for ChromaDB storage.
        ChromaDB requires all metadata values to be strings, ints, floats, or bools.
        """
        prepared = {}
        
        for key, value in metadata.items():
            if key == "page_numbers":
                # Convert list to comma-separated string
                prepared[key] = ",".join(map(str, value))
            elif key == "metrics":
                # Serialize metrics as JSON string
                prepared[key] = json.dumps(value)
            elif isinstance(value, (str, int, float, bool)):
                prepared[key] = value
            else:
                # Convert other types to string
                prepared[key] = str(value)
        
        return prepared
    
    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            where: Metadata filters (e.g., {"ticker": "AAPL", "year": 2023})
            where_document: Document content filters
        
        Returns:
            Dictionary with 'documents', 'metadatas', 'distances', 'ids'
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"]
        )
        
        # Unpack results (query returns lists of lists)
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
            "ids": results["ids"][0],
        }
    
    def search_by_company(
        self,
        query_embedding: np.ndarray,
        ticker: str,
        year: Optional[int] = None,
        section_type: Optional[str] = None,
        n_results: int = 5
    ) -> Dict:
        """
        Search within a specific company's documents.
        """
        where = {"ticker": ticker}
        
        if year:
            where["year"] = year
        if section_type:
            where["section_type"] = section_type
        
        return self.search(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where
        )
    
    def get_all_companies(self) -> List[Dict]:
        """
        Get a list of all companies in the vector store.
        """
        # Get all unique ticker/company combinations
        all_docs = self.collection.get(
            include=["metadatas"]
        )
        
        companies = {}
        for metadata in all_docs["metadatas"]:
            ticker = metadata.get("ticker")
            if ticker and ticker not in companies:
                companies[ticker] = {
                    "ticker": ticker,
                    "company_name": metadata.get("company_name"),
                    "years": set()
                }
            
            if ticker:
                year = metadata.get("year")
                if year:
                    companies[ticker]["years"].add(year)
        
        # Convert sets to sorted lists
        result = []
        for company in companies.values():
            company["years"] = sorted(list(company["years"]))
            result.append(company)
        
        return sorted(result, key=lambda x: x["ticker"])
    
    def get_company_years(self, ticker: str) -> List[int]:
        """
        Get all years available for a specific company.
        """
        results = self.collection.get(
            where={"ticker": ticker},
            include=["metadatas"]
        )
        
        years = set()
        for metadata in results["metadatas"]:
            year = metadata.get("year")
            if year:
                years.add(year)
        
        return sorted(list(years))
    
    def delete_company_year(self, ticker: str, year: int) -> int:
        """
        Delete all documents for a specific company/year.
        Returns number of documents deleted.
        """
        # Get IDs to delete
        results = self.collection.get(
            where={"ticker": ticker, "year": year},
            include=["ids"]
        )
        
        ids_to_delete = results["ids"]
        
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks for {ticker} {year}")
        
        return len(ids_to_delete)
    
    def reset_collection(self):
        """
        Delete all documents in the collection.
        WARNING: This is irreversible!
        """
        logger.warning("Resetting collection - all data will be deleted!")
        self.client.delete_collection(self.collection_name)
        self.collection = self._get_or_create_collection()
        logger.info("Collection reset complete")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        """
        total_count = self.collection.count()
        companies = self.get_all_companies()
        
        # Calculate total years across all companies
        total_years = sum(len(c["years"]) for c in companies)
        
        return {
            "total_chunks": total_count,
            "total_companies": len(companies),
            "total_company_years": total_years,
            "companies": companies,
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory),
        }


# Example usage
if __name__ == "__main__":
    # Initialize vector store
    vector_store = VectorStore(
        persist_directory="./chroma_db",
        collection_name="annual_reports"
    )
    
    # Example: Add some sample data
    sample_chunks = [
        {
            "text": "Apple reported revenue of $394 billion in fiscal 2023.",
            "embedding": np.random.rand(384),  # In real usage, use actual embeddings
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
        }
    ]
    
    # Add to vector store
    stats = vector_store.add_documents(sample_chunks)
    print("Add stats:", stats)
    
    # Get all companies
    companies = vector_store.get_all_companies()
    print(f"\nCompanies in store: {companies}")
    
    # Get statistics
    store_stats = vector_store.get_stats()
    print(f"\nVector store stats: {store_stats}")
    
    # Example search
    query_embedding = np.random.rand(384)
    results = vector_store.search_by_company(
        query_embedding=query_embedding,
        ticker="AAPL",
        year=2023,
        n_results=3
    )
    print(f"\nSearch results: {len(results['documents'])} documents found")

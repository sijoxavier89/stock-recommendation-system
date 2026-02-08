"""
PDF Processor Service
Extracts and structures text from annual reports for embedding and storage.
"""
import pdfplumber
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FinancialMetric:
    """Structured financial data extracted from reports."""
    metric_name: str
    value: float
    unit: str  # "USD", "percentage", "count"
    year: int
    context: str  # Surrounding text for context


@dataclass
class ReportSection:
    """A semantic section of the annual report."""
    section_type: str  # "executive_summary", "financials", "risks", "outlook", etc.
    title: str
    content: str
    page_numbers: List[int]
    tables: List[Dict]  # Structured table data
    metrics: List[FinancialMetric]


class PDFProcessor:
    """
    Processes annual report PDFs and extracts structured information.
    
    Key Design Decisions:
    1. Section Detection: Uses heading patterns and common report structures
    2. Financial Metrics: Extracts key numbers with context
    3. Table Preservation: Keeps table structure for later retrieval
    4. Metadata Enrichment: Tags every chunk with company, year, section type
    """
    
    # Common section headings in annual reports
    SECTION_PATTERNS = {
        "executive_summary": [
            r"executive\s+summary",
            r"letter\s+to\s+shareholders",
            r"chairman'?s\s+letter",
        ],
        "business_overview": [
            r"business\s+overview",
            r"our\s+business",
            r"company\s+overview",
        ],
        "financial_highlights": [
            r"financial\s+highlights",
            r"financial\s+summary",
            r"selected\s+financial\s+data",
        ],
        "management_discussion": [
            r"management'?s\s+discussion",
            r"md&a",
            r"financial\s+condition\s+and\s+results",
        ],
        "risk_factors": [
            r"risk\s+factors",
            r"principal\s+risks",
        ],
        "financial_statements": [
            r"consolidated\s+statements",
            r"balance\s+sheet",
            r"income\s+statement",
            r"cash\s+flow",
        ],
    }
    
    # Financial metric patterns - comprehensive set for stock analysis
    METRIC_PATTERNS = {
        # Sales metrics
        "sales": [
            r"(?:total\s+)?(?:net\s+)?sales\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
            r"(?:total\s+)?revenue[s]?\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
        ],
        "sales_growth": [
            r"(?:revenue|sales)\s+(?:growth|increase|growth\s+rate)\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
            r"(?:yoy|year[- ]over[- ]year)\s+(?:revenue|sales)\s+(?:growth|increase)\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
        ],
        
        # Receivables
        "receivables": [
            r"accounts?\s+receivables?\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
            r"trade\s+receivables?\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
        ],
        "receivables_to_sales": [
            r"receivables?\s+(?:to|/)\s+sales\s+ratio\s*[:=]?\s*([\d,]+\.?\d*)\s*%?",
            r"days\s+sales?\s+outstanding\s*[:=]?\s*([\d,]+\.?\d*)",
        ],
        "receivables_growth": [
            r"receivables?\s+(?:growth|increase)\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
        ],
        
        # Inventory
        "inventory": [
            r"(?:total\s+)?inventor(?:y|ies)\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
        ],
        "inventory_growth": [
            r"inventor(?:y|ies)\s+(?:growth|increase)\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
        ],
        
        # Profit metrics
        "operating_profit": [
            r"operating\s+(?:income|profit)\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
            r"(?:income|profit)\s+from\s+operations\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
        ],
        "net_profit": [
            r"net\s+(?:income|profit|earnings)\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
            r"profit\s+(?:after|attributable\s+to)\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
        ],
        "net_profit_growth": [
            r"net\s+(?:income|profit)\s+(?:growth|increase)\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
            r"earnings\s+growth\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
        ],
        
        # Earnings per share
        "eps": [
            r"(?:basic\s+)?(?:diluted\s+)?earnings\s+per\s+share\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)",
            r"eps\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)",
        ],
        "eps_growth": [
            r"eps\s+(?:growth|increase)\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
            r"earnings\s+per\s+share\s+(?:growth|increase)\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
        ],
        
        # Margin metrics
        "operating_profit_margin": [
            r"operating\s+(?:profit\s+)?margin\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
            r"operating\s+income\s+margin\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
        ],
        "net_profit_margin": [
            r"net\s+(?:profit\s+)?margin\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
            r"profit\s+margin\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
        ],
        
        # Efficiency ratios
        "asset_turnover": [
            r"asset\s+turnover\s+ratio\s*[:=]?\s*([\d,]+\.?\d*)",
            r"total\s+asset\s+turnover\s*[:=]?\s*([\d,]+\.?\d*)",
        ],
        "financial_leverage": [
            r"financial\s+leverage\s+ratio\s*[:=]?\s*([\d,]+\.?\d*)",
            r"equity\s+multiplier\s*[:=]?\s*([\d,]+\.?\d*)",
        ],
        
        # Return metrics
        "return_on_equity": [
            r"return\s+on\s+(?:shareholders?\s+)?equity\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
            r"roe\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
        ],
        
        # Debt metrics
        "debt_to_equity": [
            r"debt[- ]to[- ]equity\s+ratio\s*[:=]?\s*([\d,]+\.?\d*)",
            r"debt/equity\s*[:=]?\s*([\d,]+\.?\d*)",
            r"d/e\s+ratio\s*[:=]?\s*([\d,]+\.?\d*)",
        ],
        "interest_coverage": [
            r"interest\s+coverage\s+ratio\s*[:=]?\s*([\d,]+\.?\d*)",
            r"times\s+interest\s+earned\s*[:=]?\s*([\d,]+\.?\d*)",
        ],
        
        # Tax metrics
        "tax_rate": [
            r"effective\s+tax\s+rate\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
            r"income\s+tax\s+rate\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
        ],
        
        # Cash flow
        "cash_flow_from_operations": [
            r"(?:net\s+)?cash\s+(?:flow\s+)?(?:from|provided\s+by)\s+operating\s+activities\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
            r"operating\s+cash\s+flow\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
            r"cash\s+from\s+operations\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
        ],
    }
    
    def __init__(self):
        self.current_section = None
        self.sections: List[ReportSection] = []
    
    def process_pdf(
        self,
        pdf_path: Path,
        company_name: str,
        ticker: str,
        year: int
    ) -> Dict:
        """
        Main processing pipeline for a PDF annual report.
        
        Returns a dictionary with:
        - sections: List of ReportSection objects
        - metadata: Company info, year, page count
        - raw_text: Full extracted text
        - metrics: Extracted financial metrics
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        sections = []
        all_text = []
        all_metrics = []
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Total pages: {total_pages}")
            
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text
                text = page.extract_text() or ""
                all_text.append(text)
                
                # Detect sections based on headings
                detected_section = self._detect_section(text)
                
                # Extract tables from the page
                tables = page.extract_tables()
                
                # Extract financial metrics
                metrics = self._extract_metrics(text, year)
                all_metrics.extend(metrics)
                
                # Build section data
                if detected_section or page_num == 1:
                    if self.current_section:
                        sections.append(self.current_section)
                    
                    self.current_section = ReportSection(
                        section_type=detected_section or "general",
                        title=self._extract_title(text),
                        content=text,
                        page_numbers=[page_num],
                        tables=tables or [],
                        metrics=metrics,
                    )
                else:
                    # Continue current section
                    if self.current_section:
                        self.current_section.content += "\n" + text
                        self.current_section.page_numbers.append(page_num)
                        if tables:
                            self.current_section.tables.extend(tables)
                        self.current_section.metrics.extend(metrics)
            
            # Add the last section
            if self.current_section:
                sections.append(self.current_section)
        
        return {
            "sections": sections,
            "metadata": {
                "company_name": company_name,
                "ticker": ticker,
                "year": year,
                "total_pages": total_pages,
                "file_path": str(pdf_path),
            },
            "raw_text": "\n\n".join(all_text),
            "metrics": all_metrics,
        }
    
    def _detect_section(self, text: str) -> Optional[str]:
        """
        Detect which section type this page belongs to based on headings.
        """
        text_lower = text.lower()
        
        for section_type, patterns in self.SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    logger.info(f"Detected section: {section_type}")
                    return section_type
        
        return None
    
    def _extract_title(self, text: str) -> str:
        """
        Extract a title from the page (usually the first line or first heading).
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if lines:
            # Return first non-empty line, truncated to reasonable length
            return lines[0][:200]
        return "Untitled Section"
    
    def _extract_metrics(self, text: str, year: int) -> List[FinancialMetric]:
        """
        Extract financial metrics from text using regex patterns.
        """
        metrics = []
        
        for metric_name, patterns in self.METRIC_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        # Extract value and unit
                        value_str = match.group(1).replace(",", "")
                        value = float(value_str)
                        
                        # Determine unit multiplier
                        unit = match.group(2).lower() if len(match.groups()) > 1 else ""
                        if unit in ["billion", "b"]:
                            value *= 1_000_000_000
                            unit = "USD"
                        elif unit in ["million", "m"]:
                            value *= 1_000_000
                            unit = "USD"
                        elif "%" in text[match.start():match.end() + 5]:
                            unit = "percentage"
                        else:
                            unit = "USD"
                        
                        # Get surrounding context (50 chars before and after)
                        start = max(0, match.start() - 50)
                        end = min(len(text), match.end() + 50)
                        context = text[start:end].strip()
                        
                        metrics.append(FinancialMetric(
                            metric_name=metric_name,
                            value=value,
                            unit=unit,
                            year=year,
                            context=context,
                        ))
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Failed to parse metric: {e}")
                        continue
        
        return metrics
    
    def prepare_for_embedding(
        self,
        processed_data: Dict,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> List[Dict]:
        """
        Convert processed PDF data into chunks ready for embedding.
        
        Each chunk contains:
        - text: The actual text to embed
        - metadata: Company, year, section, page numbers, etc.
        
        Returns a list of dictionaries, each representing one chunk.
        """
        chunks = []
        sections = processed_data["sections"]
        metadata = processed_data["metadata"]
        
        for section in sections:
            # Split section content into chunks
            section_chunks = self._chunk_text(
                section.content,
                chunk_size=chunk_size,
                overlap=chunk_overlap
            )
            
            for i, chunk_text in enumerate(section_chunks):
                chunk_metadata = {
                    "company_name": metadata["company_name"],
                    "ticker": metadata["ticker"],
                    "year": metadata["year"],
                    "section_type": section.section_type,
                    "section_title": section.title,
                    "page_numbers": section.page_numbers,
                    "chunk_index": i,
                    "total_chunks_in_section": len(section_chunks),
                    "file_path": metadata["file_path"],
                }
                
                # Add any financial metrics found in this chunk
                chunk_metrics = [
                    m for m in section.metrics
                    if m.context in chunk_text
                ]
                
                if chunk_metrics:
                    chunk_metadata["metrics"] = [
                        {
                            "name": m.metric_name,
                            "value": m.value,
                            "unit": m.unit,
                        }
                        for m in chunk_metrics
                    ]
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                })
        
        logger.info(f"Created {len(chunks)} chunks from {len(sections)} sections")
        return chunks
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into overlapping chunks based on token count.
        
        Uses a simple word-based approximation:
        - Assume 1 token ~= 0.75 words (rough approximation)
        - Create overlapping chunks to preserve context
        """
        words = text.split()
        word_chunk_size = int(chunk_size * 0.75)
        word_overlap = int(overlap * 0.75)
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + word_chunk_size
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))
            
            # Move forward by chunk_size - overlap
            start += (word_chunk_size - word_overlap)
            
            # Stop if we've processed everything
            if end >= len(words):
                break
        
        return chunks


# Example usage
if __name__ == "__main__":
    processor = PDFProcessor()
    
    # Process a sample PDF
    pdf_path = Path("data/annual_reports/aapl/2023.pdf")
    
    processed = processor.process_pdf(
        pdf_path=pdf_path,
        company_name="Apple Inc.",
        ticker="AAPL",
        year=2023
    )
    
    # Prepare chunks for embedding
    chunks = processor.prepare_for_embedding(processed)
    
    print(f"Extracted {len(chunks)} chunks")
    print(f"Sample chunk: {chunks[0]}")

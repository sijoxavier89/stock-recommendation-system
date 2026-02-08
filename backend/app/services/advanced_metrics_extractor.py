"""
Advanced Financial Metrics Extractor
Extracts comprehensive financial metrics from annual reports including tables.
Handles both text-based extraction and structured table parsing.
"""
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FinancialMetric:
    """Enhanced financial metric with calculation support."""
    metric_name: str
    value: float
    unit: str  # "USD", "percentage", "ratio", "days"
    year: int
    context: str
    source: str  # "text" or "table"
    confidence: float  # 0.0 to 1.0
    calculation_method: Optional[str] = None  # How it was derived


class AdvancedMetricsExtractor:
    """
    Comprehensive financial metrics extraction from annual reports.
    
    Capabilities:
    1. Text-based regex extraction
    2. Table parsing for financial statements
    3. Ratio calculation from base metrics
    4. Multi-year trend detection
    5. Validation and confidence scoring
    """
    
    # Complete metric patterns for your requested metrics
    METRIC_PATTERNS = {
        # Sales metrics
        "sales": {
            "patterns": [
                r"(?:total\s+)?(?:net\s+)?sales\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
                r"(?:total\s+)?revenue[s]?\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
            ],
            "unit": "USD",
            "aliases": ["revenue", "total_revenue", "net_sales"],
        },
        "sales_growth": {
            "patterns": [
                r"(?:revenue|sales)\s+(?:growth|increase)\s*[:=]?\s*([-]?[\d,]+\.?\d*)\s*%",
                r"(?:yoy|year[- ]over[- ]year)\s+(?:revenue|sales)\s+(?:growth|increase)\s*[:=]?\s*([-]?[\d,]+\.?\d*)\s*%",
            ],
            "unit": "percentage",
            "calculated_from": ["sales"],  # Can be calculated if multi-year sales available
        },
        
        # Receivables
        "receivables": {
            "patterns": [
                r"accounts?\s+receivables?\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
                r"trade\s+receivables?\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
            ],
            "unit": "USD",
            "table_keywords": ["accounts receivable", "trade receivables"],
        },
        "receivables_to_sales": {
            "patterns": [
                r"receivables?\s+(?:to|/)\s+sales\s+ratio\s*[:=]?\s*([\d,]+\.?\d*)\s*%?",
                r"days\s+sales?\s+outstanding\s*[:=]?\s*([\d,]+\.?\d*)",
            ],
            "unit": "ratio",
            "calculated_from": ["receivables", "sales"],
            "formula": "receivables / sales",
        },
        "receivables_growth": {
            "patterns": [
                r"receivables?\s+(?:growth|increase)\s*[:=]?\s*([-]?[\d,]+\.?\d*)\s*%",
            ],
            "unit": "percentage",
            "calculated_from": ["receivables"],
        },
        
        # Inventory
        "inventory": {
            "patterns": [
                r"(?:total\s+)?inventor(?:y|ies)\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
            ],
            "unit": "USD",
            "table_keywords": ["inventory", "inventories"],
        },
        "inventory_growth": {
            "patterns": [
                r"inventor(?:y|ies)\s+(?:growth|increase)\s*[:=]?\s*([-]?[\d,]+\.?\d*)\s*%",
            ],
            "unit": "percentage",
            "calculated_from": ["inventory"],
        },
        
        # Profit metrics
        "operating_profit": {
            "patterns": [
                r"operating\s+(?:income|profit)\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
                r"(?:income|profit)\s+from\s+operations\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
            ],
            "unit": "USD",
            "table_keywords": ["operating income", "income from operations"],
        },
        "net_profit": {
            "patterns": [
                r"net\s+(?:income|profit|earnings)\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
                r"profit\s+(?:after\s+tax|attributable\s+to)\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
            ],
            "unit": "USD",
            "table_keywords": ["net income", "net profit", "net earnings"],
        },
        "net_profit_growth": {
            "patterns": [
                r"net\s+(?:income|profit)\s+(?:growth|increase)\s*[:=]?\s*([-]?[\d,]+\.?\d*)\s*%",
            ],
            "unit": "percentage",
            "calculated_from": ["net_profit"],
        },
        
        # Earnings per share
        "eps": {
            "patterns": [
                r"(?:basic\s+)?(?:diluted\s+)?earnings\s+per\s+share\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)",
                r"eps\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)",
            ],
            "unit": "USD",
            "table_keywords": ["earnings per share", "eps"],
        },
        "eps_growth": {
            "patterns": [
                r"eps\s+(?:growth|increase)\s*[:=]?\s*([-]?[\d,]+\.?\d*)\s*%",
            ],
            "unit": "percentage",
            "calculated_from": ["eps"],
        },
        
        # Margins
        "operating_profit_margin": {
            "patterns": [
                r"operating\s+(?:profit\s+)?margin\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
            ],
            "unit": "percentage",
            "calculated_from": ["operating_profit", "sales"],
            "formula": "(operating_profit / sales) * 100",
        },
        "net_profit_margin": {
            "patterns": [
                r"net\s+(?:profit\s+)?margin\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
            ],
            "unit": "percentage",
            "calculated_from": ["net_profit", "sales"],
            "formula": "(net_profit / sales) * 100",
        },
        
        # Efficiency ratios
        "asset_turnover": {
            "patterns": [
                r"asset\s+turnover\s+ratio\s*[:=]?\s*([\d,]+\.?\d*)",
            ],
            "unit": "ratio",
            "calculated_from": ["sales", "total_assets"],
            "formula": "sales / total_assets",
        },
        "financial_leverage": {
            "patterns": [
                r"financial\s+leverage\s+ratio\s*[:=]?\s*([\d,]+\.?\d*)",
                r"equity\s+multiplier\s*[:=]?\s*([\d,]+\.?\d*)",
            ],
            "unit": "ratio",
            "calculated_from": ["total_assets", "shareholders_equity"],
            "formula": "total_assets / shareholders_equity",
        },
        
        # Return metrics
        "return_on_equity": {
            "patterns": [
                r"return\s+on\s+(?:shareholders?\s+)?equity\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
                r"roe\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
            ],
            "unit": "percentage",
            "calculated_from": ["net_profit", "shareholders_equity"],
            "formula": "(net_profit / shareholders_equity) * 100",
        },
        
        # Debt metrics
        "debt_to_equity": {
            "patterns": [
                r"debt[- ]to[- ]equity\s+ratio\s*[:=]?\s*([\d,]+\.?\d*)",
                r"debt/equity\s*[:=]?\s*([\d,]+\.?\d*)",
            ],
            "unit": "ratio",
            "calculated_from": ["total_debt", "shareholders_equity"],
            "formula": "total_debt / shareholders_equity",
        },
        "interest_coverage": {
            "patterns": [
                r"interest\s+coverage\s+ratio\s*[:=]?\s*([\d,]+\.?\d*)",
                r"times\s+interest\s+earned\s*[:=]?\s*([\d,]+\.?\d*)",
            ],
            "unit": "ratio",
            "calculated_from": ["operating_profit", "interest_expense"],
            "formula": "operating_profit / interest_expense",
        },
        
        # Tax
        "tax_rate": {
            "patterns": [
                r"effective\s+tax\s+rate\s*[:=]?\s*([\d,]+\.?\d*)\s*%",
            ],
            "unit": "percentage",
        },
        
        # Cash flow
        "cash_flow_from_operations": {
            "patterns": [
                r"(?:net\s+)?cash\s+(?:flow\s+)?(?:from|provided\s+by)\s+operating\s+activities\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
                r"operating\s+cash\s+flow\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
            ],
            "unit": "USD",
            "table_keywords": ["cash from operating activities", "operating cash flow"],
        },
        
        # Supporting metrics (needed for calculations)
        "total_assets": {
            "patterns": [
                r"total\s+assets\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
            ],
            "unit": "USD",
            "table_keywords": ["total assets"],
        },
        "shareholders_equity": {
            "patterns": [
                r"(?:total\s+)?shareholders?\s+equity\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
                r"stockholders?\s+equity\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
            ],
            "unit": "USD",
            "table_keywords": ["shareholders equity", "stockholders equity"],
        },
        "total_debt": {
            "patterns": [
                r"total\s+debt\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
                r"long[- ]term\s+debt\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
            ],
            "unit": "USD",
            "table_keywords": ["total debt", "long-term debt"],
        },
        "interest_expense": {
            "patterns": [
                r"interest\s+expense\s*[:=]?\s*\$?\s*([\d,]+\.?\d*)\s*(million|billion|m|b)?",
            ],
            "unit": "USD",
            "table_keywords": ["interest expense"],
        },
    }
    
    def __init__(self):
        self.extracted_metrics: Dict[str, List[FinancialMetric]] = {}
    
    def extract_from_text(self, text: str, year: int) -> List[FinancialMetric]:
        """Extract metrics from text using regex patterns."""
        metrics = []
        
        for metric_name, config in self.METRIC_PATTERNS.items():
            patterns = config.get("patterns", [])
            unit = config.get("unit", "unknown")
            
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        value_str = match.group(1).replace(",", "")
                        value = float(value_str)
                        
                        # Apply multiplier if present
                        if len(match.groups()) > 1 and match.group(2):
                            multiplier_str = match.group(2).lower()
                            if multiplier_str in ["billion", "b"]:
                                value *= 1_000_000_000
                            elif multiplier_str in ["million", "m"]:
                                value *= 1_000_000
                        
                        # Get context
                        start = max(0, match.start() - 100)
                        end = min(len(text), match.end() + 100)
                        context = text[start:end].strip()
                        
                        metric = FinancialMetric(
                            metric_name=metric_name,
                            value=value,
                            unit=unit if unit != "unknown" else ("USD" if match.group(0).find("$") >= 0 else "count"),
                            year=year,
                            context=context,
                            source="text",
                            confidence=0.8,  # Text extraction has moderate confidence
                        )
                        
                        metrics.append(metric)
                        
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Failed to parse metric {metric_name}: {e}")
                        continue
        
        return metrics
    
    def extract_from_table(self, table: List[List[str]], year: int) -> List[FinancialMetric]:
        """
        Extract metrics from a structured table.
        
        Args:
            table: List of rows, where each row is a list of cell values
            year: Fiscal year
        
        Returns:
            List of extracted metrics
        """
        metrics = []
        
        if not table or len(table) < 2:
            return metrics
        
        # Common table structures:
        # 1. Row headers with year columns: [Metric, 2023, 2022, 2021]
        # 2. Column headers with metrics: rows contain values
        
        # Try to identify year columns
        header_row = table[0]
        year_columns = self._identify_year_columns(header_row, year)
        
        # Extract metrics from each row
        for row_idx, row in enumerate(table[1:], start=1):
            if not row or len(row) < 2:
                continue
            
            # First column often contains metric name
            metric_label = str(row[0]).strip().lower()
            
            # Match against our known metrics
            for metric_name, config in self.METRIC_PATTERNS.items():
                keywords = config.get("table_keywords", []) + config.get("aliases", [])
                
                # Check if this row matches our metric
                if any(keyword in metric_label for keyword in keywords):
                    # Extract values from year columns
                    for col_idx, col_year in year_columns.items():
                        if col_idx < len(row):
                            try:
                                value = self._parse_table_value(row[col_idx])
                                if value is not None:
                                    metric = FinancialMetric(
                                        metric_name=metric_name,
                                        value=value,
                                        unit=config.get("unit", "USD"),
                                        year=col_year,
                                        context=f"From table row: {row[0]}",
                                        source="table",
                                        confidence=0.95,  # Table data has high confidence
                                    )
                                    metrics.append(metric)
                            except Exception as e:
                                logger.debug(f"Failed to parse table value: {e}")
        
        return metrics
    
    def _identify_year_columns(self, header_row: List[str], target_year: int) -> Dict[int, int]:
        """Identify which columns contain which years."""
        year_columns = {}
        
        for idx, cell in enumerate(header_row):
            cell_str = str(cell).strip()
            
            # Look for 4-digit years
            year_match = re.search(r'\b(20\d{2})\b', cell_str)
            if year_match:
                year = int(year_match.group(1))
                year_columns[idx] = year
        
        # If no years found in header, assume first data column is target year
        if not year_columns and len(header_row) > 1:
            year_columns[1] = target_year
        
        return year_columns
    
    def _parse_table_value(self, cell: str) -> Optional[float]:
        """Parse a value from a table cell."""
        cell_str = str(cell).strip()
        
        # Remove common formatting
        cell_str = cell_str.replace("$", "").replace(",", "").replace("(", "-").replace(")", "")
        
        # Handle percentages
        is_percentage = "%" in cell_str
        cell_str = cell_str.replace("%", "")
        
        # Try to parse as float
        try:
            value = float(cell_str)
            return value
        except ValueError:
            return None
    
    def calculate_derived_metrics(
        self,
        base_metrics: Dict[str, FinancialMetric],
        year: int
    ) -> List[FinancialMetric]:
        """
        Calculate derived metrics from base metrics.
        
        For example:
        - ROE = Net Profit / Shareholders Equity
        - Operating Margin = Operating Profit / Sales
        """
        derived = []
        
        for metric_name, config in self.METRIC_PATTERNS.items():
            if "calculated_from" not in config:
                continue
            
            required_metrics = config["calculated_from"]
            formula = config.get("formula")
            
            # Check if we have all required base metrics
            if all(req in base_metrics for req in required_metrics):
                try:
                    # Calculate the metric
                    if metric_name == "operating_profit_margin":
                        value = (base_metrics["operating_profit"].value / base_metrics["sales"].value) * 100
                    elif metric_name == "net_profit_margin":
                        value = (base_metrics["net_profit"].value / base_metrics["sales"].value) * 100
                    elif metric_name == "return_on_equity":
                        value = (base_metrics["net_profit"].value / base_metrics["shareholders_equity"].value) * 100
                    elif metric_name == "debt_to_equity":
                        value = base_metrics["total_debt"].value / base_metrics["shareholders_equity"].value
                    elif metric_name == "receivables_to_sales":
                        value = base_metrics["receivables"].value / base_metrics["sales"].value
                    elif metric_name == "asset_turnover":
                        value = base_metrics["sales"].value / base_metrics["total_assets"].value
                    elif metric_name == "financial_leverage":
                        value = base_metrics["total_assets"].value / base_metrics["shareholders_equity"].value
                    elif metric_name == "interest_coverage":
                        value = base_metrics["operating_profit"].value / base_metrics["interest_expense"].value
                    else:
                        continue
                    
                    metric = FinancialMetric(
                        metric_name=metric_name,
                        value=value,
                        unit=config.get("unit", "ratio"),
                        year=year,
                        context=f"Calculated from: {', '.join(required_metrics)}",
                        source="calculated",
                        confidence=0.9,
                        calculation_method=formula,
                    )
                    
                    derived.append(metric)
                    
                except (ZeroDivisionError, KeyError) as e:
                    logger.debug(f"Cannot calculate {metric_name}: {e}")
        
        return derived
    
    def calculate_growth_metrics(
        self,
        current_year_metrics: Dict[str, FinancialMetric],
        prior_year_metrics: Dict[str, FinancialMetric]
    ) -> List[FinancialMetric]:
        """
        Calculate year-over-year growth metrics.
        
        For metrics like sales_growth, eps_growth, etc.
        """
        growth_metrics = []
        
        growth_mappings = {
            "sales": "sales_growth",
            "net_profit": "net_profit_growth",
            "eps": "eps_growth",
            "receivables": "receivables_growth",
            "inventory": "inventory_growth",
        }
        
        for base_metric, growth_metric in growth_mappings.items():
            if base_metric in current_year_metrics and base_metric in prior_year_metrics:
                current = current_year_metrics[base_metric]
                prior = prior_year_metrics[base_metric]
                
                if prior.value != 0:
                    growth_rate = ((current.value - prior.value) / prior.value) * 100
                    
                    metric = FinancialMetric(
                        metric_name=growth_metric,
                        value=growth_rate,
                        unit="percentage",
                        year=current.year,
                        context=f"YoY growth from {prior.year} to {current.year}",
                        source="calculated",
                        confidence=0.95,
                        calculation_method=f"({current.value} - {prior.value}) / {prior.value} * 100",
                    )
                    
                    growth_metrics.append(metric)
        
        return growth_metrics


# Example usage
if __name__ == "__main__":
    extractor = AdvancedMetricsExtractor()
    
    # Test text extraction
    sample_text = """
    The company reported total revenue of $394.3 billion for fiscal 2023,
    representing a year-over-year growth of 9.2%. Net income was $99.8 billion,
    with earnings per share of $6.13. The operating margin was 30.1% and
    net profit margin reached 25.3%. Return on equity was strong at 147%.
    """
    
    metrics = extractor.extract_from_text(sample_text, year=2023)
    
    print(f"Extracted {len(metrics)} metrics from text:")
    for m in metrics:
        print(f"  {m.metric_name}: {m.value} {m.unit} (confidence: {m.confidence})")
    
    # Test table extraction
    sample_table = [
        ["Financial Metric", "2023", "2022", "2021"],
        ["Total Revenue ($M)", "394,328", "365,817", "274,515"],
        ["Net Income ($M)", "99,803", "94,680", "86,802"],
        ["Earnings Per Share ($)", "6.13", "5.67", "5.11"],
        ["Operating Margin (%)", "30.1", "30.3", "29.8"],
    ]
    
    table_metrics = extractor.extract_from_table(sample_table, year=2023)
    
    print(f"\nExtracted {len(table_metrics)} metrics from table:")
    for m in table_metrics:
        print(f"  {m.metric_name} ({m.year}): {m.value} {m.unit}")

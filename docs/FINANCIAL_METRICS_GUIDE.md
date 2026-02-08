# Financial Metrics Extraction Guide

## Overview

This system extracts **21 key financial metrics** from annual reports to enable quantitative stock analysis. This guide explains what each metric means, how we extract it, and how to use it for recommendations.

---

## Complete Metrics List

### 1. Sales Metrics

#### **Sales (Revenue)**
- **What it is**: Total revenue generated from operations
- **Formula**: Sum of all revenue streams
- **Extraction**: Text patterns + Income Statement table
- **Unit**: USD (millions/billions)
- **Why it matters**: Top-line growth indicator
- **Good value**: Growing year-over-year

**Example extraction:**
```
Text: "Total revenue of $394.3 billion"
Table: Row "Total Revenue" → $394,328M
```

#### **Sales Growth**
- **What it is**: Year-over-year revenue increase
- **Formula**: `(Sales_current - Sales_prior) / Sales_prior * 100`
- **Extraction**: Text patterns OR calculated from multi-year sales
- **Unit**: Percentage
- **Why it matters**: Measures business expansion
- **Good value**: >10% for growth companies, >5% for mature companies

---

### 2. Working Capital Metrics

#### **Receivables (Accounts Receivable)**
- **What it is**: Money owed by customers
- **Extraction**: Balance Sheet table
- **Unit**: USD
- **Why it matters**: Liquidity and collection efficiency
- **Warning sign**: Growing faster than sales

#### **Receivables/Sales Ratio**
- **What it is**: Receivables as proportion of sales
- **Formula**: `Receivables / Sales`
- **Unit**: Ratio or Days Sales Outstanding
- **Why it matters**: Collection efficiency
- **Good value**: <0.15 (15% of sales) or <45 days

#### **Receivables Growth**
- **Formula**: YoY change in receivables
- **Warning sign**: Growing faster than sales growth (potential collection issues)

#### **Inventory**
- **What it is**: Goods available for sale
- **Extraction**: Balance Sheet
- **Why it matters**: Operational efficiency
- **Warning sign**: Growing faster than sales (overstocking)

#### **Inventory Growth**
- **Formula**: YoY change in inventory
- **Good sign**: Growth matches or lags sales growth

---

### 3. Profitability Metrics

#### **Operating Profit (Operating Income)**
- **What it is**: Profit from core operations before interest/tax
- **Formula**: `Revenue - Operating Expenses`
- **Extraction**: Income Statement
- **Unit**: USD
- **Why it matters**: Core business profitability
- **Good value**: Positive and growing

#### **Net Profit (Net Income)**
- **What it is**: Bottom-line profit after all expenses
- **Formula**: `Operating Profit - Interest - Tax`
- **Extraction**: Income Statement (most important line)
- **Unit**: USD
- **Why it matters**: Actual earnings available to shareholders
- **Good value**: Positive and growing faster than revenue

#### **Net Profit Growth**
- **Formula**: YoY change in net profit
- **Why it matters**: Earnings momentum
- **Good value**: >15% for growth stocks

---

### 4. Per-Share Metrics

#### **EPS (Earnings Per Share)**
- **What it is**: Profit allocated to each share
- **Formula**: `Net Income / Shares Outstanding`
- **Extraction**: Income Statement or EPS section
- **Unit**: USD per share
- **Why it matters**: Stock valuation (used in P/E ratio)
- **Good value**: Growing consistently

#### **EPS Growth**
- **Formula**: YoY change in EPS
- **Why it matters**: Direct impact on stock price
- **Good value**: >15% consistently

---

### 5. Margin Metrics

#### **Operating Profit Margin**
- **What it is**: Operating efficiency
- **Formula**: `(Operating Profit / Sales) * 100`
- **Unit**: Percentage
- **Why it matters**: How much profit from each dollar of sales
- **Good value**: >15% (varies by industry)
- **Best companies**: 25%+ (Apple, Microsoft)

#### **Net Profit Margin**
- **What it is**: After-tax profitability
- **Formula**: `(Net Profit / Sales) * 100`
- **Unit**: Percentage
- **Why it matters**: Overall efficiency
- **Good value**: >10% (software: 20%+, retail: 5%+)

---

### 6. Efficiency Ratios

#### **Asset Turnover**
- **What it is**: Revenue generated per dollar of assets
- **Formula**: `Sales / Total Assets`
- **Unit**: Ratio
- **Why it matters**: Asset utilization efficiency
- **Good value**: >1.0 (higher is better)

#### **Financial Leverage**
- **What it is**: Degree of debt usage
- **Formula**: `Total Assets / Shareholders Equity`
- **Unit**: Ratio (Equity Multiplier)
- **Why it matters**: Risk assessment
- **Good value**: 1.5-2.5 (varies by industry)

---

### 7. Return Metrics

#### **Return on Equity (ROE)**
- **What it is**: Profit generated from shareholders' investment
- **Formula**: `(Net Profit / Shareholders Equity) * 100`
- **Unit**: Percentage
- **Why it matters**: Key profitability metric
- **Good value**: >15% (excellent: >20%)
- **Top performers**: 30%+ (Apple: 147%)

**This is THE most important metric for stock recommendations.**

---

### 8. Debt & Solvency Metrics

#### **Debt to Equity Ratio**
- **What it is**: Leverage level
- **Formula**: `Total Debt / Shareholders Equity`
- **Unit**: Ratio
- **Why it matters**: Financial risk
- **Good value**: <1.0 (conservative), <2.0 (acceptable)
- **Warning**: >3.0 (high risk)

#### **Interest Coverage**
- **What it is**: Ability to pay interest
- **Formula**: `Operating Profit / Interest Expense`
- **Unit**: Ratio (times)
- **Why it matters**: Debt sustainability
- **Good value**: >5x (safe), >10x (very safe)
- **Warning**: <2x (distress risk)

---

### 9. Tax Metrics

#### **Tax Rate (Effective Tax Rate)**
- **What it is**: Actual tax percentage paid
- **Formula**: `Income Tax / Pre-tax Income * 100`
- **Unit**: Percentage
- **Why it matters**: Tax efficiency
- **Typical value**: 15-25% (varies by country)

---

### 10. Cash Flow Metrics

#### **Cash Flow from Operations**
- **What it is**: Cash generated from core business
- **Extraction**: Cash Flow Statement
- **Unit**: USD
- **Why it matters**: Real cash generation (not accounting profit)
- **Good value**: Positive and > Net Profit
- **Warning**: Negative (burning cash)

**Golden Rule**: Operating Cash Flow > Net Income is healthy

---

## Extraction Strategy

### Method 1: Text-Based Extraction

```python
from advanced_metrics_extractor import AdvancedMetricsExtractor

extractor = AdvancedMetricsExtractor()

# Extract from narrative sections
text = section_content  # From PDF processor
metrics = extractor.extract_from_text(text, year=2023)
```

**Pros:**
- Catches metrics in MD&A sections
- Gets commentary and context

**Cons:**
- Lower confidence (narrative can be ambiguous)
- May miss exact values

### Method 2: Table-Based Extraction

```python
# Extract from financial statement tables
table = pdf_page.extract_tables()[0]  # From pdfplumber
metrics = extractor.extract_from_table(table, year=2023)
```

**Pros:**
- Higher confidence (structured data)
- Captures multi-year comparisons
- More accurate values

**Cons:**
- Requires proper table detection
- Tables may span multiple pages

### Method 3: Calculated Metrics

```python
# Calculate derived metrics from base metrics
base_metrics = {
    "net_profit": metric_obj,
    "shareholders_equity": metric_obj,
}

derived = extractor.calculate_derived_metrics(base_metrics, year=2023)
# Returns: ROE, margins, ratios, etc.
```

**Pros:**
- Fills in missing metrics
- Ensures consistency
- Allows custom calculations

### Method 4: Multi-Year Growth

```python
# Calculate growth rates
growth = extractor.calculate_growth_metrics(
    current_year_metrics=metrics_2023,
    prior_year_metrics=metrics_2022
)
# Returns: sales_growth, eps_growth, etc.
```

---

## Integration with Pipeline

### In PDF Processor

```python
# Modified pdf_processor.py
from advanced_metrics_extractor import AdvancedMetricsExtractor

class PDFProcessor:
    def __init__(self):
        self.metrics_extractor = AdvancedMetricsExtractor()
    
    def process_pdf(self, pdf_path, company_name, ticker, year):
        # ... existing code ...
        
        all_metrics = []
        
        for page in pdf.pages:
            text = page.extract_text()
            tables = page.extract_tables()
            
            # Extract from text
            text_metrics = self.metrics_extractor.extract_from_text(text, year)
            all_metrics.extend(text_metrics)
            
            # Extract from tables
            for table in tables:
                table_metrics = self.metrics_extractor.extract_from_table(table, year)
                all_metrics.extend(table_metrics)
        
        # Deduplicate and prioritize (table > text)
        unique_metrics = self._deduplicate_metrics(all_metrics)
        
        # Calculate derived metrics
        base_metrics = {m.metric_name: m for m in unique_metrics}
        derived = self.metrics_extractor.calculate_derived_metrics(base_metrics, year)
        
        return {
            "metrics": unique_metrics + derived,
            # ... rest of data
        }
```

---

## Storing Metrics in Vector DB

### Enhanced Chunk Metadata

```python
chunk_metadata = {
    "company_name": "Apple Inc.",
    "ticker": "AAPL",
    "year": 2023,
    "section_type": "financial_highlights",
    
    # Add structured metrics
    "metrics": {
        "sales": 394328000000,
        "sales_growth": 9.2,
        "net_profit": 99803000000,
        "net_profit_margin": 25.3,
        "eps": 6.13,
        "eps_growth": 8.1,
        "return_on_equity": 147.0,
        "operating_profit_margin": 30.1,
        "debt_to_equity": 1.98,
        "cash_flow_from_operations": 110543000000,
        # ... all 21 metrics
    }
}
```

### Benefits

1. **Filtering**: Find high-ROE companies
```python
results = vector_store.search(
    where={
        "metrics.return_on_equity": {"$gte": 20},  # ROE > 20%
        "metrics.sales_growth": {"$gte": 10}       # Growth > 10%
    }
)
```

2. **Quantitative Context**: LLM gets exact numbers
```python
prompt = f"""
Company: {metadata['company_name']}
ROE: {metadata['metrics']['return_on_equity']}%
Net Profit Margin: {metadata['metrics']['net_profit_margin']}%

Is this a good investment?
"""
```

3. **Ranking**: Sort by performance
```python
companies = sorted(all_companies, 
                   key=lambda c: c['metrics']['return_on_equity'], 
                   reverse=True)
```

---

## Using Metrics for Recommendations

### Example: ROE-Based Ranking

```python
def get_top_roe_companies(min_roe=15, min_growth=10):
    """Find companies with high ROE and growth."""
    companies = vector_store.get_all_companies()
    
    recommendations = []
    for company in companies:
        # Get latest year metrics
        latest_year = max(company['years'])
        metrics = get_company_metrics(company['ticker'], latest_year)
        
        roe = metrics.get('return_on_equity', 0)
        growth = metrics.get('sales_growth', 0)
        
        if roe >= min_roe and growth >= min_growth:
            recommendations.append({
                'ticker': company['ticker'],
                'name': company['company_name'],
                'roe': roe,
                'growth': growth,
                'score': roe * 0.7 + growth * 0.3  # Weighted score
            })
    
    return sorted(recommendations, key=lambda x: x['score'], reverse=True)
```

### Example: Multi-Metric Screening

```python
def screen_companies(criteria):
    """
    Screen companies based on multiple criteria.
    
    Example criteria:
    {
        'sales_growth': {'min': 10},
        'net_profit_margin': {'min': 15},
        'return_on_equity': {'min': 20},
        'debt_to_equity': {'max': 2.0},
        'interest_coverage': {'min': 5},
    }
    """
    passing_companies = []
    
    for company in all_companies:
        metrics = get_company_metrics(company['ticker'], 2023)
        
        passes = True
        for metric_name, conditions in criteria.items():
            value = metrics.get(metric_name)
            if value is None:
                passes = False
                break
            
            if 'min' in conditions and value < conditions['min']:
                passes = False
                break
            if 'max' in conditions and value > conditions['max']:
                passes = False
                break
        
        if passes:
            passing_companies.append(company)
    
    return passing_companies
```

---

## Metrics Priority for Recommendations

### Tier 1: Must-Have Metrics (Core Fundamentals)
1. **Sales** - Business size
2. **Sales Growth** - Momentum
3. **Net Profit** - Profitability
4. **EPS** - Per-share value
5. **Return on Equity** - Efficiency (MOST IMPORTANT)

### Tier 2: Important Metrics (Quality Indicators)
6. **Operating Profit Margin** - Operational efficiency
7. **Net Profit Margin** - Overall profitability
8. **Cash Flow from Operations** - Cash generation
9. **Debt to Equity** - Financial health
10. **Interest Coverage** - Debt safety

### Tier 3: Nice-to-Have Metrics (Operational Details)
11. Receivables/Sales
12. Inventory Growth
13. Asset Turnover
14. Financial Leverage
15. Tax Rate

---

## Validation & Quality Checks

### Sanity Checks

```python
def validate_metrics(metrics):
    """Check for logical inconsistencies."""
    issues = []
    
    # Check: Net Profit should be less than Sales
    if metrics.get('net_profit', 0) > metrics.get('sales', 0):
        issues.append("Net profit exceeds sales (impossible)")
    
    # Check: Net margin should be less than 100%
    if metrics.get('net_profit_margin', 0) > 100:
        issues.append("Net margin > 100% (error)")
    
    # Check: Receivables shouldn't be >50% of sales
    if metrics.get('receivables_to_sales', 0) > 0.5:
        issues.append("Receivables >50% of sales (collection issue?)")
    
    # Check: ROE consistency
    if 'return_on_equity' in metrics and 'net_profit' in metrics:
        calculated_roe = (metrics['net_profit'] / metrics['shareholders_equity']) * 100
        if abs(calculated_roe - metrics['return_on_equity']) > 1:
            issues.append("ROE calculation mismatch")
    
    return issues
```

---

## Summary: Metrics-Driven Recommendations

Your system will now be able to:

1. ✅ Extract 21 key financial metrics automatically
2. ✅ Calculate derived metrics (margins, ratios, growth)
3. ✅ Store metrics as structured data in vector DB
4. ✅ Filter companies by quantitative criteria
5. ✅ Rank companies using custom scoring
6. ✅ Provide LLM with exact numbers (no hallucination)
7. ✅ Generate recommendations based on fundamentals

**This is what sets your system apart from generic RAG systems!**

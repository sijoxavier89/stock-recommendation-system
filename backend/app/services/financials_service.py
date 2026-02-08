"""Financial metrics service.

Tries to fetch annual financial statements via yfinance and compute common ratios.
If yfinance isn't available or a field is missing, values will be `None`.

This is a pragmatic, best-effort implementation — field names vary across providers so
we attempt several common index names.
"""
from typing import Optional, Dict


def _find_column_for_year(df, year: int):
    if df is None or df.empty:
        return None
    for col in df.columns:
        # pandas Timestamp has .year
        try:
            if getattr(col, "year", None) == int(year):
                return col
        except Exception:
            continue
    # fallback: try string match
    for col in df.columns:
        if str(col).startswith(str(year)):
            return col
    return None


def _get_value(df, year: int, candidates):
    if df is None or df.empty:
        return None
    col = _find_column_for_year(df, year)
    if col is None:
        return None
    for name in candidates:
        if name in df.index:
            try:
                v = df.at[name, col]
                if v is None:
                    continue
                return float(v)
            except Exception:
                continue
    # try case-insensitive match
    idx = {str(i).lower(): i for i in df.index}
    for name in candidates:
        key = name.lower()
        if key in idx:
            try:
                return float(df.at[idx[key], col])
            except Exception:
                continue
    return None


def _pct_change(curr: Optional[float], prev: Optional[float]) -> Optional[float]:
    if curr is None or prev is None:
        return None
    try:
        if prev == 0:
            return None
        return (curr - prev) / abs(prev)
    except Exception:
        return None


def get_financial_metrics(ticker: str, year: int) -> Dict[str, Optional[float]]:
    """Return the requested financial metrics for `ticker` at `year`.

    This function attempts to read financial statements via yfinance. Missing
    values will be returned as None.
    """
    try:
        import yfinance as yf
    except Exception:
        # yfinance not installed or failed to import
        return {k: None for k in [
            "Sales", "Sales Growth", "Receivables", "Receivables/Sales", "Receivables Growth",
            "Inventory", "Inventory Growth", "Operating Profit", "Net profit", "Net profit Growth",
            "EPS", "EPS Growth", "Operating Profit Margin", "Net profit Margin", "Asset Turnover",
            "Financial Leverage", "Return on Equity", "Debt to equity ratio", "Interest Coverage",
            "Tax rate", "Cash Flow from operation"
        ]}

    tk = yf.Ticker(ticker)
    fin = tk.financials  # income statement (annual)
    bal = tk.balance_sheet
    cf = tk.cashflow

    # common name candidates
    sales_names = ["Total Revenue", "TotalRevenue", "Revenue", "Revenues", "Sales", "totalRevenue"]
    operating_names = ["Operating Income", "OperatingIncome", "Operating Profit", "Operating profit", "OperatingIncomeLoss", "OperatingIncomeOrLoss"]
    net_income_names = ["Net Income", "NetIncome", "Net income", "Net Income Available to Common Stockholders", "NetIncomeLoss"]
    receivables_names = ["Net receivables", "Net Receivables", "Accounts Receivable, Net", "Accounts Receivable", "Trade Receivables"]
    inventory_names = ["Inventory", "Inventories", "Inventory, net"]
    interest_names = ["Interest Expense", "InterestExpense"]
    tax_names = ["Income Tax Expense", "IncomeTaxExpense", "Provision for Income Taxes"]
    total_assets_names = ["Total Assets", "TotalAssets"]
    total_equity_names = ["Total Stockholders' Equity", "Total stockholders' equity", "Total Stockholder Equity", "Total shareholders' equity", "Total Equity"]
    total_liab_names = ["Total Liab", "Total Liabilities", "Total liabilities", "Total Liab Net Minority Interest"]
    cashflow_op_names = ["Total Cash From Operating Activities", "Net cash provided by operating activities", "Cash Flow from Operations", "Operating Cash Flow"]

    def get_from(df, names):
        return _get_value(df, year, names)

    sales = get_from(fin, sales_names)
    sales_prev = _get_value(fin, year - 1, sales_names)

    receivables = get_from(bal, receivables_names)
    receivables_prev = _get_value(bal, receivables_names + ["Accounts Receivable"], year - 1) if receivables is not None else _get_value(bal, receivables_names, year - 1)

    inventory = get_from(bal, inventory_names)
    inventory_prev = _get_value(bal, inventory_names, year - 1)

    operating_profit = get_from(fin, operating_names) or get_from(fin, ["Operating Income (Loss)"]) or get_from(fin, ["OperatingIncomeLoss"])
    net_profit = get_from(fin, net_income_names)
    net_prev = _get_value(fin, year - 1, net_income_names)

    # EPS: try earnings per share from tk.earnings or info
    eps = None
    eps_prev = None
    try:
        # some tickers provide diluted EPS in 'earnings' or 'earnings_history'
        earnings = tk.earnings  # DataFrame with Year and Earnings (not EPS)
        info = tk.info or {}
        if info:
            eps = info.get("trailingEps")
        # attempt to compute EPS as Net Income / sharesOutstanding
        if eps is None:
            shares = info.get("sharesOutstanding")
            if net_profit is not None and shares:
                eps = net_profit / shares
    except Exception:
        eps = None

    # margins and ratios
    operating_margin = None
    net_margin = None
    if sales and operating_profit is not None and sales != 0:
        operating_margin = operating_profit / sales
    if sales and net_profit is not None and sales != 0:
        net_margin = net_profit / sales

    asset_turnover = None
    total_assets = get_from(bal, total_assets_names)
    if total_assets and sales:
        try:
            asset_turnover = sales / total_assets
        except Exception:
            asset_turnover = None

    financial_leverage = None
    total_equity = get_from(bal, total_equity_names)
    if total_assets and total_equity:
        try:
            financial_leverage = total_assets / total_equity
        except Exception:
            financial_leverage = None

    return_on_equity = None
    if net_profit is not None and total_equity:
        try:
            return_on_equity = net_profit / total_equity
        except Exception:
            return_on_equity = None

    debt_to_equity = None
    total_liab = get_from(bal, total_liab_names)
    if total_liab is not None and total_equity:
        try:
            debt_to_equity = total_liab / total_equity
        except Exception:
            debt_to_equity = None

    interest = get_from(fin, interest_names) or get_from(fin, ["Interest Expense"])
    interest_coverage = None
    if interest and operating_profit:
        try:
            interest_coverage = operating_profit / abs(interest) if interest != 0 else None
        except Exception:
            interest_coverage = None

    tax = get_from(fin, tax_names)
    tax_rate = None
    if tax is not None and net_profit not in (None, 0):
        try:
            tax_rate = tax / net_profit
        except Exception:
            tax_rate = None

    cashflow_op = get_from(cf, cashflow_op_names)

    metrics = {
        "Sales": sales,
        "Sales Growth": _pct_change(sales, sales_prev),
        "Receivables": receivables,
        "Receivables/Sales": (receivables / sales) if (receivables is not None and sales) else None,
        "Receivables Growth": _pct_change(receivables, receivables_prev),
        "Inventory": inventory,
        "Inventory Growth": _pct_change(inventory, inventory_prev),
        "Operating Profit": operating_profit,
        "Net profit": net_profit,
        "Net profit Growth": _pct_change(net_profit, net_prev),
        "EPS": eps,
        "EPS Growth": None,  # EPS growth requires year-by-year EPS which may not be available
        "Operating Profit Margin": operating_margin,
        "Net profit Margin": net_margin,
        "Asset Turnover": asset_turnover,
        "Financial Leverage": financial_leverage,
        "Return on Equity": return_on_equity,
        "Debt to equity ratio": debt_to_equity,
        "Interest Coverage": interest_coverage,
        "Tax rate": tax_rate,
        "Cash Flow from operation": cashflow_op,
    }

    return metrics

export interface Company {
  ticker: string;
  company_name: string;
  years: number[];
  chunks: number;
}

export interface FinancialMetrics {
  sales?: number;
  sales_growth?: number;
  net_profit?: number;
  net_profit_margin?: number;
  eps?: number;
  eps_growth?: number;
  return_on_equity?: number;
  debt_to_equity?: number;
  operating_profit_margin?: number;
  cash_flow_from_operations?: number;
}
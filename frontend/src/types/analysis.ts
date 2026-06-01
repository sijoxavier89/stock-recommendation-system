export interface AnalysisRequest {
  query: string;
  company_ticker?: string;
  year?: number;
  section_type?: string;
}

export interface ComparisonRequest {
  company_tickers: string[];
  metric_focus?: string;
  year?: number;
}

export interface RecommendationRequest {
  company_ticker: string;
  investment_style: 'growth' | 'value' | 'balanced';
  year?: number;
}

export interface RetrievedChunk {
  text: string;
  company_name: string;
  ticker: string;
  year: number;
  section_type: string;
  similarity_score: number;
  page_numbers: number[];
}

export interface AnalysisResponse {
  answer: string;
  retrieved_chunks: RetrievedChunk[];
  metrics_used?: Record<string, any>;
  confidence: number;
  reasoning?: string;
  timestamp: string;
}

export interface QueryHistory {
  id: string;
  query: string;
  timestamp: Date;
  confidence: number;
}

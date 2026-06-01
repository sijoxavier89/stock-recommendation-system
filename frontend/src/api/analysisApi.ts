import apiClient from './client';
import {
  AnalysisRequest,
  AnalysisResponse,
  ComparisonRequest,
  RecommendationRequest,
} from '@/types/analysis';

export const analysisApi = {
  // Ask a question
  query: async (data: AnalysisRequest): Promise<AnalysisResponse> => {
    const response = await apiClient.post('/api/analysis/query', data);
    return response.data;
  },

  // Compare companies
  compare: async (data: ComparisonRequest): Promise<AnalysisResponse> => {
    const response = await apiClient.post('/api/analysis/compare', data);
    return response.data;
  },

  // Get recommendation
  recommend: async (data: RecommendationRequest): Promise<AnalysisResponse> => {
    const response = await apiClient.post('/api/analysis/recommend', data);
    return response.data;
  },

  // Health check
  health: async () => {
    const response = await apiClient.get('/api/analysis/health');
    return response.data;
  },
};
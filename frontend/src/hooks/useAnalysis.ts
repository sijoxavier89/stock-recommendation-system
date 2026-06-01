import { useMutation } from '@tanstack/react-query';
import { analysisApi } from '@/api';
import { AnalysisRequest } from '@/types/analysis';

export const useAnalysis = () => {
  return useMutation({
    mutationFn: (data: AnalysisRequest) => analysisApi.query(data),
    onSuccess: (data) => {
      console.log('Analysis successful:', data);
    },
    onError: (error) => {
      console.error('Analysis failed:', error);
    },
  });
};

export const useComparison = () => {
  return useMutation({
    mutationFn: analysisApi.compare,
  });
};

export const useRecommendation = () => {
  return useMutation({
    mutationFn: analysisApi.recommend,
  });
};
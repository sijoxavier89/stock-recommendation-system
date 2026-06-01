import { create } from 'zustand';
import { AnalysisResponse, QueryHistory } from '@/types/analysis';

interface AnalysisState {
  currentResult: AnalysisResponse | null;
  queryHistory: QueryHistory[];
  isLoading: boolean;
  error: string | null;
  
  setCurrentResult: (result: AnalysisResponse | null) => void;
  addToHistory: (query: string, confidence: number) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  clearHistory: () => void;
}

export const useAnalysisStore = create<AnalysisState>((set) => ({
  currentResult: null,
  queryHistory: [],
  isLoading: false,
  error: null,

  setCurrentResult: (result) => set({ currentResult: result }),
  
  addToHistory: (query, confidence) =>
    set((state) => ({
      queryHistory: [
        {
          id: Date.now().toString(),
          query,
          timestamp: new Date(),
          confidence,
        },
        ...state.queryHistory.slice(0, 9), // Keep last 10
      ],
    })),

  setLoading: (loading) => set({ isLoading: loading }),
  setError: (error) => set({ error }),
  clearHistory: () => set({ queryHistory: [] }),
}));
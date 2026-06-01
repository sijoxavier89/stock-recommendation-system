import { useState } from 'react';
import { Container, Typography, Box, CircularProgress } from '@mui/material';
import { QueryInput } from '@/components/Analysis/QueryInput';
import { ResultDisplay } from '@/components/Analysis/ResultDisplay';
import { analysisApi } from '@/api';
import { useAnalysisStore } from '@/stores/analysisStore';
import { AnalysisResponse } from '@/types/analysis';

export const Analysis = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Mock companies data - replace with actual API call
  const companies = [
    { ticker: 'AAPL', years: [2021, 2022, 2023] },
    { ticker: 'MSFT', years: [2021, 2022, 2023] },
    { ticker: 'GOOGL', years: [2021, 2022, 2023] },
  ];

  const handleQuery = async (query: string, ticker?: string, year?: number) => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await analysisApi.query({
        query,
        company_ticker: ticker,
        year,
      });
      
      setResult(response);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to get analysis');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Ask Questions
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Ask questions about companies and get AI-powered insights from annual reports.
        </Typography>

        <QueryInput
          onSubmit={handleQuery}
          isLoading={isLoading}
          companies={companies}
        />

        {isLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
            <CircularProgress />
          </Box>
        )}

        {error && (
          <Typography color="error" sx={{ mt: 2 }}>
            {error}
          </Typography>
        )}

        {result && <ResultDisplay result={result} />}
      </Box>
    </Container>
  );
};
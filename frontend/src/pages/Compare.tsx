import { useState } from 'react';
import {
  Container,
  Typography,
  Box,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Stack,
  Paper,
  CircularProgress,
} from '@mui/material';
import { analysisApi } from '@/api';
import { AnalysisResponse } from '@/types/analysis';
import { ResultDisplay } from '@/components/Analysis/ResultDisplay';

export const Compare = () => {
  const [selectedTickers, setSelectedTickers] = useState<string[]>([]);
  const [metricFocus, setMetricFocus] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);

  // Mock companies
  const availableCompanies = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'];
  const availableMetrics = ['ROE', 'Revenue Growth', 'Profit Margin', 'Debt to Equity'];

  const handleCompare = async () => {
    if (selectedTickers.length < 2) return;

    setIsLoading(true);
    try {
      const response = await analysisApi.compare({
        company_tickers: selectedTickers,
        metric_focus: metricFocus || undefined,
      });
      setResult(response);
    } catch (err) {
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" gutterBottom>
          Compare Companies
        </Typography>

        <Paper sx={{ p: 3, mt: 3 }}>
          <Stack spacing={3}>
            <FormControl fullWidth>
              <InputLabel>Select Companies (2-5)</InputLabel>
              <Select
                multiple
                value={selectedTickers}
                onChange={(e) => setSelectedTickers(e.target.value as string[])}
                renderValue={(selected) => (
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {selected.map((value) => (
                      <Chip key={value} label={value} />
                    ))}
                  </Box>
                )}
              >
                {availableCompanies.map((ticker) => (
                  <MenuItem key={ticker} value={ticker}>
                    {ticker}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl fullWidth>
              <InputLabel>Focus Metric (Optional)</InputLabel>
              <Select
                value={metricFocus}
                onChange={(e) => setMetricFocus(e.target.value)}
              >
                <MenuItem value="">
                  <em>All Metrics</em>
                </MenuItem>
                {availableMetrics.map((metric) => (
                  <MenuItem key={metric} value={metric}>
                    {metric}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Button
              variant="contained"
              size="large"
              onClick={handleCompare}
              disabled={selectedTickers.length < 2 || isLoading}
            >
              {isLoading ? 'Comparing...' : 'Compare Companies'}
            </Button>
          </Stack>
        </Paper>

        {isLoading && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
            <CircularProgress />
          </Box>
        )}

        {result && <ResultDisplay result={result} />}
      </Box>
    </Container>
  );
};
import { useState } from 'react';
import {
  Paper,
  TextField,
  Button,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Stack,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';

interface QueryInputProps {
  onSubmit: (query: string, ticker?: string, year?: number) => void;
  isLoading: boolean;
  companies: Array<{ ticker: string; years: number[] }>;
}

export const QueryInput = ({ onSubmit, isLoading, companies }: QueryInputProps) => {
  const [query, setQuery] = useState('');
  const [selectedTicker, setSelectedTicker] = useState('');
  const [selectedYear, setSelectedYear] = useState<number | ''>('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      onSubmit(
        query,
        selectedTicker || undefined,
        selectedYear || undefined
      );
    }
  };

  const selectedCompany = companies.find((c) => c.ticker === selectedTicker);

  return (
    <Paper sx={{ p: 3 }}>
      <form onSubmit={handleSubmit}>
        <Stack spacing={2}>
          <TextField
            fullWidth
            label="Ask a question about companies"
            placeholder="e.g., What was Apple's revenue growth in 2023?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            multiline
            rows={3}
            disabled={isLoading}
          />

          <Box sx={{ display: 'flex', gap: 2 }}>
            <FormControl sx={{ minWidth: 200 }}>
              <InputLabel>Company (Optional)</InputLabel>
              <Select
                value={selectedTicker}
                onChange={(e) => {
                  setSelectedTicker(e.target.value);
                  setSelectedYear('');
                }}
                label="Company (Optional)"
                disabled={isLoading}
              >
                <MenuItem value="">
                  <em>All Companies</em>
                </MenuItem>
                {companies.map((company) => (
                  <MenuItem key={company.ticker} value={company.ticker}>
                    {company.ticker}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <FormControl sx={{ minWidth: 150 }}>
              <InputLabel>Year (Optional)</InputLabel>
              <Select
                value={selectedYear}
                onChange={(e) => setSelectedYear(e.target.value as number)}
                label="Year (Optional)"
                disabled={isLoading || !selectedCompany}
              >
                <MenuItem value="">
                  <em>All Years</em>
                </MenuItem>
                {selectedCompany?.years.map((year) => (
                  <MenuItem key={year} value={year}>
                    {year}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            <Button
              type="submit"
              variant="contained"
              startIcon={<SearchIcon />}
              disabled={isLoading || !query.trim()}
              sx={{ minWidth: 120 }}
            >
              {isLoading ? 'Analyzing...' : 'Ask'}
            </Button>
          </Box>
        </Stack>
      </form>
    </Paper>
  );
};
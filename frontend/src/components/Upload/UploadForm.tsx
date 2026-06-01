import { useState } from 'react';
import {
  Paper,
  TextField,
  Button,
  Box,
  Typography,
  Stack,
  Alert,
} from '@mui/material';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import { uploadApi, UploadProgress } from '@/api';

export const UploadForm = () => {
  const [file, setFile] = useState<File | null>(null);
  const [ticker, setTicker] = useState('');
  const [companyName, setCompanyName] = useState('');
  const [year, setYear] = useState('');
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!file || !ticker || !companyName || !year) {
      setError('Please fill all fields');
      return;
    }

    try {
      setError(null);
      await uploadApi.uploadReport(
        file,
        ticker.toUpperCase(),
        companyName,
        parseInt(year),
        setProgress
      );
      
      setProgress({ status: 'success', progress: 100, message: 'Upload complete!' });
      
      // Reset form
      setTimeout(() => {
        setFile(null);
        setTicker('');
        setCompanyName('');
        setYear('');
        setProgress(null);
      }, 2000);
      
    } catch (err: any) {
      setError(err.message || 'Upload failed');
      setProgress(null);
    }
  };

  return (
    <Paper sx={{ p: 3 }}>
      <Typography variant="h6" gutterBottom>
        Upload Annual Report
      </Typography>

      <form onSubmit={handleSubmit}>
        <Stack spacing={3}>
          <Button
            variant="outlined"
            component="label"
            startIcon={<UploadFileIcon />}
            fullWidth
          >
            {file ? file.name : 'Choose PDF File'}
            <input
              type="file"
              hidden
              accept=".pdf"
              onChange={handleFileChange}
            />
          </Button>

          <TextField
            label="Ticker Symbol"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="e.g., AAPL"
            required
          />

          <TextField
            label="Company Name"
            value={companyName}
            onChange={(e) => setCompanyName(e.target.value)}
            placeholder="e.g., Apple Inc."
            required
          />

          <TextField
            label="Year"
            type="number"
            value={year}
            onChange={(e) => setYear(e.target.value)}
            placeholder="e.g., 2023"
            required
            inputProps={{ min: 2000, max: new Date().getFullYear() }}
          />

          {error && <Alert severity="error">{error}</Alert>}
          
          {progress && (
            <Alert severity={progress.status === 'success' ? 'success' : 'info'}>
              {progress.message}
            </Alert>
          )}

          <Button
            type="submit"
            variant="contained"
            size="large"
            disabled={!file || progress?.status === 'uploading'}
          >
            {progress?.status === 'uploading' ? 'Uploading...' : 'Upload'}
          </Button>
        </Stack>
      </form>
    </Paper>
  );
};
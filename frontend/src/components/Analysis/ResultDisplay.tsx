import {
  Paper,
  Typography,
  Box,
  Chip,
  LinearProgress,
  Alert,
} from '@mui/material';
import { AnalysisResponse } from '@/types/analysis';
import { SourceCitations } from './SourceCitations';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import WarningIcon from '@mui/icons-material/Warning';

interface ResultDisplayProps {
  result: AnalysisResponse;
}

export const ResultDisplay = ({ result }: ResultDisplayProps) => {
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'success';
    if (confidence >= 0.6) return 'warning';
    return 'error';
  };

  const getConfidenceIcon = (confidence: number) => {
    if (confidence >= 0.7) return <CheckCircleIcon />;
    return <WarningIcon />;
  };

  return (
    <Paper sx={{ p: 3, mt: 3 }}>
      <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Typography variant="h6">Analysis Result</Typography>
        <Chip
          icon={getConfidenceIcon(result.confidence)}
          label={`${(result.confidence * 100).toFixed(0)}% Confidence`}
          color={getConfidenceColor(result.confidence)}
        />
      </Box>

      <LinearProgress
        variant="determinate"
        value={result.confidence * 100}
        color={getConfidenceColor(result.confidence)}
        sx={{ mb: 2 }}
      />

      {result.confidence < 0.6 && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          Low confidence score. Results may not be comprehensive. Try refining your question or specifying a company.
        </Alert>
      )}

      <Typography variant="body1" sx={{ mb: 3, whiteSpace: 'pre-wrap' }}>
        {result.answer}
      </Typography>

      {result.reasoning && (
        <Box sx={{ mb: 3, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
          <Typography variant="subtitle2" color="text.secondary" gutterBottom>
            Reasoning:
          </Typography>
          <Typography variant="body2">{result.reasoning}</Typography>
        </Box>
      )}

      {result.metrics_used && Object.keys(result.metrics_used).length > 0 && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Key Metrics Used:
          </Typography>
          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {Object.entries(result.metrics_used).map(([key, value]) => (
              <Chip
                key={key}
                label={`${key}: ${typeof value === 'number' ? value.toLocaleString() : value}`}
                size="small"
                variant="outlined"
              />
            ))}
          </Box>
        </Box>
      )}

      <SourceCitations chunks={result.retrieved_chunks} />
    </Paper>
  );
};
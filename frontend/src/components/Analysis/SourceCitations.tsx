import {
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Typography,
  Chip,
  Box,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import { RetrievedChunk } from '@/types/analysis';

interface SourceCitationsProps {
  chunks: RetrievedChunk[];
}

export const SourceCitations = ({ chunks }: SourceCitationsProps) => {
  if (!chunks || chunks.length === 0) return null;

  return (
    <Box>
      <Typography variant="subtitle2" gutterBottom>
        Sources ({chunks.length}):
      </Typography>
      
      {chunks.map((chunk, index) => (
        <Accordion key={index}>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Box sx={{ display: 'flex', gap: 1, alignItems: 'center', width: '100%' }}>
              <Typography sx={{ flexGrow: 1 }}>
                {chunk.company_name} ({chunk.ticker}) - {chunk.year}
              </Typography>
              <Chip
                label={`${(chunk.similarity_score * 100).toFixed(0)}% relevant`}
                size="small"
                color={chunk.similarity_score > 0.8 ? 'success' : 'default'}
              />
            </Box>
          </AccordionSummary>
          
          <AccordionDetails>
            <Box>
              <Typography variant="caption" color="text.secondary" gutterBottom>
                Section: {chunk.section_type} | Pages: {chunk.page_numbers.join(', ')}
              </Typography>
              <Typography variant="body2" sx={{ mt: 1 }}>
                {chunk.text}
              </Typography>
            </Box>
          </AccordionDetails>
        </Accordion>
      ))}
    </Box>
  );
};
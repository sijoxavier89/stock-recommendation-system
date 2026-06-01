import { Container, Typography, Box } from '@mui/material';
import { UploadForm } from '@/components/Upload/UploadForm';

export const Upload = () => {
  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Upload Annual Report
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Add new company annual reports to the system for analysis.
        </Typography>

        <UploadForm />
      </Box>
    </Container>
  );
};
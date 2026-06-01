import { Container, Typography, Box, Button, Grid, Paper } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import SearchIcon from '@mui/icons-material/Search';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import UploadFileIcon from '@mui/icons-material/UploadFile';

export const Home = () => {
  const navigate = useNavigate();

  const features = [
    {
      title: 'Ask Questions',
      description: 'Query financial data using natural language',
      icon: <SearchIcon sx={{ fontSize: 60 }} />,
      action: () => navigate('/analysis'),
      color: '#1976d2',
    },
    {
      title: 'Compare Companies',
      description: 'Side-by-side comparison of multiple companies',
      icon: <CompareArrowsIcon sx={{ fontSize: 60 }} />,
      action: () => navigate('/compare'),
      color: '#2e7d32',
    },
    {
      title: 'Upload Reports',
      description: 'Add new annual reports to the system',
      icon: <UploadFileIcon sx={{ fontSize: 60 }} />,
      action: () => navigate('/upload'),
      color: '#ed6c02',
    },
  ];

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 8, textAlign: 'center' }}>
        <Typography variant="h2" component="h1" gutterBottom>
          Stock Recommendation System
        </Typography>
        <Typography variant="h5" color="text.secondary" paragraph>
          AI-powered analysis of annual reports using RAG
        </Typography>
      </Box>

      <Grid container spacing={4} sx={{ mt: 4 }}>
        {features.map((feature) => (
          <Grid item xs={12} md={4} key={feature.title}>
            <Paper
              sx={{
                p: 4,
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'translateY(-8px)',
                  boxShadow: 4,
                },
              }}
              onClick={feature.action}
            >
              <Box sx={{ color: feature.color, mb: 2 }}>{feature.icon}</Box>
              <Typography variant="h5" gutterBottom>
                {feature.title}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {feature.description}
              </Typography>
              <Button
                variant="contained"
                sx={{ mt: 2, bgcolor: feature.color }}
                onClick={feature.action}
              >
                Get Started
              </Button>
            </Paper>
          </Grid>
        ))}
      </Grid>
    </Container>
  );
};
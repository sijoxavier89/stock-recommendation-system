import { Container, Typography, Box, Grid, Paper, Chip } from '@mui/material';

export const Companies = () => {
  // Mock data - replace with actual API call
  const companies = [
    { ticker: 'AAPL', name: 'Apple Inc.', years: [2021, 2022, 2023], chunks: 234 },
    { ticker: 'MSFT', name: 'Microsoft Corporation', years: [2021, 2022, 2023], chunks: 189 },
    { ticker: 'GOOGL', name: 'Alphabet Inc.', years: [2022, 2023], chunks: 156 },
  ];

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" gutterBottom>
          Indexed Companies
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Companies currently available for analysis
        </Typography>

        <Grid container spacing={3} sx={{ mt: 2 }}>
          {companies.map((company) => (
            <Grid item xs={12} md={4} key={company.ticker}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  {company.ticker}
                </Typography>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  {company.name}
                </Typography>
                
                <Box sx={{ mt: 2 }}>
                  <Typography variant="caption" color="text.secondary">
                    Years Available:
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 0.5, mt: 0.5, flexWrap: 'wrap' }}>
                    {company.years.map((year) => (
                      <Chip key={year} label={year} size="small" />
                    ))}
                  </Box>
                </Box>

                <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
                  {company.chunks} document chunks
                </Typography>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </Box>
    </Container>
  );
};
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import { Link } from 'react-router-dom';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';

export const Navbar = () => {
  return (
    <AppBar position="static">
      <Toolbar>
        <TrendingUpIcon sx={{ mr: 2 }} />
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          Stock Analyzer
        </Typography>
        
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button color="inherit" component={Link} to="/">
            Home
          </Button>
          <Button color="inherit" component={Link} to="/analysis">
            Analysis
          </Button>
          <Button color="inherit" component={Link} to="/compare">
            Compare
          </Button>
          <Button color="inherit" component={Link} to="/upload">
            Upload
          </Button>
          <Button color="inherit" component={Link} to="/companies">
            Companies
          </Button>
        </Box>
      </Toolbar>
    </AppBar>
  );
};
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import { CssBaseline } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { theme } from './styles/theme';
import { Layout } from './components/Layout/Layout';
import { Home } from './pages/Home';
import { Analysis } from './pages/Analysis';
import { Compare } from './pages/Compare';
import { Upload } from './pages/Upload';
import { Companies } from './pages/Companies';

// Create React Query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Layout />}>
              <Route index element={<Home />} />
              <Route path="analysis" element={<Analysis />} />
              <Route path="compare" element={<Compare />} />
              <Route path="upload" element={<Upload />} />
              <Route path="companies" element={<Companies />} />
            </Route>
          </Routes>
        </BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
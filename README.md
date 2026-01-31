# Stock Recommendation System

An AI-powered stock recommendation engine that analyzes companies based on their historical annual reports using **RAG (Retrieval-Augmented Generation)**, a **vector database**, and **LLM-based insights**. The system ingests annual report PDFs, embeds them into a local vector store, and surfaces actionable recommendations through a modern React dashboard.

> ⚠️ **Disclaimer**: This system is for educational and research purposes only. It does not constitute financial advice. Always consult a qualified financial advisor before making investment decisions.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Environment Setup](#environment-setup)
6. [Backend Setup](#backend-setup)
7. [Frontend Setup](#frontend-setup)
8. [How It Works](#how-it-works)
9. [API Reference](#api-reference)
10. [MCP Server Integration](#mcp-server-integration)
11. [Adding Annual Reports](#adding-annual-reports)
12. [Configuration](#configuration)
13. [Free Tools & Alternatives](#free-tools--alternatives)
14. [Roadmap](#roadmap)
15. [Contributing](#contributing)
16. [License](#license)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        React Frontend                        │
│   ┌──────────┐  ┌─────────────┐  ┌──────────┐  ┌─────────┐  │
│   │ Filter   │  │   Upload    │  │ News     │  │ Dash-   │  │
│   │ Panel    │  │   Reports   │  │ Feed     │  │ board   │  │
│   └────┬─────┘  └──────┬──────┘  └────┬─────┘  └────┬────┘  │
└────────┼───────────────┼─────────────┼─────────────┼────────┘
         │               │             │             │
         └───────────────┴─────────────┴─────────────┘
                         │  REST API (CORS)
┌────────────────────────▼─────────────────────────────────────┐
│                    Python Backend (FastAPI)                    │
│                                                              │
│   ┌─────────────┐   ┌──────────────┐   ┌────────────────┐   │
│   │ PDF         │   │ RAG          │   │ LLM            │   │
│   │ Processor   │──▶│ Pipeline     │──▶│ Service        │   │
│   └─────────────┘   └──────┬───────┘   └────────────────┘   │
│                            │                                  │
│   ┌─────────────┐   ┌──────▼───────┐   ┌────────────────┐   │
│   │ News        │   │ Embedding    │   │ MCP Servers    │   │
│   │ Service     │   │ Service      │   │ (Financial/    │   │
│   └─────────────┘   └──────┬───────┘   │  News Data)    │   │
│                            │           └────────────────┘   │
└────────────────────────────┼─────────────────────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌──────────────┐ ┌───────────┐ ┌─────────────┐
     │  ChromaDB    │ │ Local PDF │ │  External   │
     │ Vector Store │ │  Storage  │ │   APIs      │
     └──────────────┘ └───────────┘ └─────────────┘
```

### Data Flow

1. **Ingestion**: Annual report PDFs are uploaded or placed in the local `data/annual_reports/` directory.
2. **Processing**: The PDF processor extracts text, splits it into semantic chunks, and generates embeddings using a local sentence-transformer model.
3. **Storage**: Embeddings and metadata are persisted in ChromaDB (local, no server required).
4. **Query**: When a user requests an analysis or recommendation, the user's query is embedded, the most relevant document chunks are retrieved via similarity search, and a prompt is assembled with those chunks as context.
5. **Generation**: The LLM generates insights, comparisons, or recommendations based on the retrieved context.
6. **Response**: Results are returned to the React frontend for display.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Frontend Framework | React 18 + TypeScript | Component-based UI, strong typing |
| State Management | Zustand | Lightweight, minimal boilerplate |
| Charts | Recharts | Free, React-native charting |
| Backend Framework | FastAPI | Async, auto-docs, fast |
| LLM | Ollama (local) / OpenAI API | Ollama is free and runs locally |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Free, runs locally, good quality |
| Vector DB | ChromaDB | Free, embedded, zero config |
| PDF Parsing | pdfplumber | Reliable text + table extraction |
| News | NewsAPI / Alpha Vantage | Free tiers available |
| Financial Data | yfinance | Free Yahoo Finance wrapper |
| MCP | Custom MCP servers | Extend LLM tool use capabilities |

---

## Project Structure

```
stock-recommendation-system/
│
├── .env                              # Environment variables (gitignored)
├── .gitignore
├── docker-compose.yml                # Optional: orchestrate backend services
├── README.md
│
├── frontend/                         # React Application
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── api/                      # API layer (axios calls)
│   │   │   ├── companyApi.ts
│   │   │   ├── uploadApi.ts
│   │   │   ├── newsApi.ts
│   │   │   └── analysisApi.ts
│   │   ├── components/
│   │   │   ├── Layout/               # Navbar, Sidebar
│   │   │   ├── Filter/               # Company filter panel
│   │   │   ├── Upload/               # Report upload + progress
│   │   │   ├── Dashboard/            # Cards, charts, recommendations
│   │   │   └── News/                 # News feed
│   │   ├── hooks/                    # Custom hooks (useCompanies, etc.)
│   │   ├── stores/                   # Zustand global state
│   │   ├── types/                    # TypeScript interfaces
│   │   ├── utils/                    # Helpers (formatters)
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
│
├── backend/                          # Python Backend
│   ├── app/
│   │   ├── main.py                   # FastAPI entry point
│   │   ├── config.py                 # Pydantic settings from .env
│   │   ├── dependencies.py           # DI (shared service instances)
│   │   ├── routes/                   # Route handlers
│   │   │   ├── companies.py
│   │   │   ├── upload.py
│   │   │   ├── analysis.py
│   │   │   └── news.py
│   │   ├── services/                 # Core business logic
│   │   │   ├── llm_service.py
│   │   │   ├── embedding_service.py
│   │   │   ├── rag_service.py
│   │   │   ├── pdf_processor.py
│   │   │   ├── vector_store.py
│   │   │   ├── company_service.py
│   │   │   └── news_service.py
│   │   ├── models/                   # Pydantic request/response models
│   │   │   ├── company.py
│   │   │   └── report.py
│   │   ├── mcp/                      # MCP server definitions
│   │   │   ├── financial_data.py
│   │   │   └── news_data.py
│   │   ├── prompts/                  # LLM prompt templates
│   │   │   ├── analysis_prompt.py
│   │   │   └── recommendation_prompt.py
│   │   └── utils/
│   │       └── chunking.py           # Text chunking strategies
│   ├── data/
│   │   └── annual_reports/           # PDF storage (gitignored contents)
│   │       └── .gitkeep
│   ├── chroma_db/                    # Auto-generated vector store
│   │   └── .gitkeep
│   ├── requirements.txt
│   └── .env.example
│
└── docs/
    ├── architecture.md
    └── api-docs.md
```

---

## Prerequisites

- **Python** 3.10+ (`python --version`)
- **Node.js** 18+ (`node --version`)
- **npm** 9+ (`npm --version`)
- **Ollama** installed (for local LLM) — [ollama.ai](https://ollama.ai)
  - Pull a model after install: `ollama pull mistral`
- **Git**

---

## Environment Setup

Copy the example env file and fill in your values:

```bash
cp backend/.env.example .env
```

### `.env` Reference

```ini
# ─── LLM Configuration ────────────────────────────────────────
LLM_PROVIDER=ollama                # "ollama" (local) or "openai"
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral               # Model pulled via `ollama pull`

# Only required if LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# ─── Embedding Model ──────────────────────────────────────────
EMBEDDING_MODEL=all-MiniLM-L6-v2   # sentence-transformers model name

# ─── ChromaDB ─────────────────────────────────────────────────
CHROMA_DB_PATH=./data/chroma_db    # Local persistence path

# ─── PDF Storage ──────────────────────────────────────────────
REPORTS_DIR=./data/annual_reports

# ─── News API ─────────────────────────────────────────────────
NEWSAPI_KEY=your_newsapi_key       # From newsapi.org (free tier)
ALPHA_VANTAGE_KEY=your_av_key      # From alphavantage.co (free tier)

# ─── Frontend ─────────────────────────────────────────────────
VITE_API_BASE_URL=http://localhost:8000
```

---

## Backend Setup

```bash
# 1. Navigate to backend directory
cd backend

# 2. Create and activate a virtual environment
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Ollama in a separate terminal (if using local LLM)
ollama serve

# 5. Run the FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be live at `http://localhost:8000`. Interactive docs are available at `http://localhost:8000/docs`.

---

## Frontend Setup

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Start the development server
npm run dev
```

The React app will be live at `http://localhost:5173`.

---

## How It Works

### 1. Uploading an Annual Report

The user uploads a PDF through the React dashboard. The frontend sends the file along with metadata (company name, ticker, fiscal year) to the backend.

The backend then runs the **ingestion pipeline**:

```
PDF File
  │
  ▼
Text Extraction (pdfplumber)
  │
  ▼
Semantic Chunking (overlap-aware splits ~500 tokens)
  │
  ▼
Embedding Generation (sentence-transformers, runs locally)
  │
  ▼
ChromaDB Storage (with metadata: company, year, section)
```

### 2. Requesting a Recommendation or Analysis

When a user asks for analysis on a company (e.g., "How did Apple's revenue trend over the last 3 years?"), the system runs the **RAG pipeline**:

```
User Query
  │
  ▼
Embed the query (same model used during ingestion)
  │
  ▼
Similarity Search against ChromaDB (top-k relevant chunks)
  │
  ▼
Assemble Prompt (system prompt + retrieved chunks + user query)
  │
  ▼
Send to LLM (Ollama local or OpenAI)
  │
  ▼
Return structured response to frontend
```

### 3. Filtering Companies

The filter panel queries the backend for companies already indexed in the vector store. Filters are applied on stored metadata (sector, year, financial metrics extracted during ingestion).

### 4. News Updates

The news feed calls external APIs (NewsAPI, Alpha Vantage) via the backend to surface the latest headlines for tracked companies. Results are cached server-side to stay within free-tier rate limits.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/companies` | List all indexed companies. Supports query params: `sector`, `min_revenue`, `max_revenue`, `year` |
| `POST` | `/api/upload-report` | Upload an annual report PDF. Multipart form: `file`, `company`, `ticker`, `year` |
| `POST` | `/api/analyze` | Run RAG analysis. Body: `{ "company": "AAPL", "query": "Revenue trend" }` |
| `GET` | `/api/news/{ticker}` | Fetch latest news for a given ticker symbol |
| `GET` | `/api/companies/{ticker}/summary` | Get a cached AI-generated summary for a company |
| `GET` | `/health` | Health check endpoint |

---

## MCP Server Integration

The `backend/app/mcp/` directory contains custom **Model Context Protocol** servers that extend the LLM's capabilities with tool use.

### `financial_data.py` — Financial Data MCP Server

Provides the LLM with tools to:
- Look up current stock prices (via `yfinance`)
- Retrieve historical price data
- Fetch market cap and basic fundamentals

### `news_data.py` — News MCP Server

Provides the LLM with tools to:
- Search for company news by ticker
- Retrieve recent SEC filings summaries

### How MCP Fits In

When the LLM needs external data it cannot answer from the retrieved document chunks alone, it can invoke an MCP tool. The backend handles the tool call, fetches real data, and returns the result to the LLM to incorporate into its response.

```
LLM decides it needs current stock price
  │
  ▼
Emits a tool_call for `get_stock_price(ticker="AAPL")`
  │
  ▼
Backend MCP handler calls yfinance
  │
  ▼
Returns price data to LLM
  │
  ▼
LLM incorporates data into final response
```

---

## Adding Annual Reports

Reports can be added in two ways:

**Option A — Via the UI (recommended)**

Use the upload panel in the React dashboard. Fill in the company name, ticker, and fiscal year, then drag and drop or select the PDF.

**Option B — Manually (bulk load)**

Place PDFs in the following structure inside `backend/data/annual_reports/`:

```
annual_reports/
├── aapl/
│   ├── 2022.pdf
│   └── 2023.pdf
├── tsla/
│   ├── 2022.pdf
│   └── 2023.pdf
└── msft/
    └── 2023.pdf
```

Then hit the backend endpoint to trigger bulk indexing:

```bash
curl -X POST http://localhost:8000/api/index-all
```

This scans the directory, processes every PDF, and populates the vector store.

---

## Configuration

| Setting | File | Default | Description |
|---|---|---|---|
| LLM Provider | `.env` | `ollama` | `ollama` for local, `openai` for API |
| Local Model | `.env` | `mistral` | Any model installed via `ollama pull` |
| Chunk Size | `utils/chunking.py` | `500` tokens | Size of each document chunk |
| Chunk Overlap | `utils/chunking.py` | `50` tokens | Overlap between consecutive chunks |
| Top-K Retrieval | `services/rag_service.py` | `5` | Number of chunks retrieved per query |
| News Cache TTL | `services/news_service.py` | `3600`s | How long news results are cached |

---

## Free Tools & Alternatives

This project is designed to run with **zero cost** using locally hosted tools. Below is a reference for swapping components if needed.

| Component | Free Option (Default) | Paid Alternative |
|---|---|---|
| LLM | Ollama + Mistral / Llama 3 | OpenAI GPT-4, Anthropic Claude |
| Embeddings | `all-MiniLM-L6-v2` (local) | OpenAI `ada-002` |
| Vector DB | ChromaDB (embedded) | Pinecone, Weaviate |
| News | NewsAPI free tier (100 req/day) | NewsAPI paid plan |
| Financial Data | yfinance (unlimited) | Alpha Vantage paid, Bloomberg |
| PDF Parsing | pdfplumber (open source) | Adobe PDF API |

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** | Core backend: FastAPI, PDF ingestion, ChromaDB, embeddings | Planned |
| **Phase 2** | RAG pipeline: chunking, retrieval, LLM integration, prompt templates | Planned |
| **Phase 3** | React frontend: dashboard, filters, upload, charts | Planned |
| **Phase 4** | Intelligence layer: recommendations, news feed, comparative analysis | Planned |
| **Phase 5** | MCP servers, caching, performance optimization, export features | Planned |

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add feature description"`
4. Push to your branch: `git push origin feature/your-feature`
5. Open a Pull Request.

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

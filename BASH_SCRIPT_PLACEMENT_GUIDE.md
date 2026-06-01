# Bash Script Placement & Service Startup Guide

Best practices for organizing startup scripts in your project.

---

## 📍 **Best Places to Add Startup Scripts**

### **Option 1: Root Level (Recommended for Small Projects)**

```
stock-recommendation-system/
├── start.sh              # ← Main startup script
├── stop.sh               # ← Stop all services
├── restart.sh            # ← Restart services
├── backend/
├── frontend/
└── README.md
```

**Pros:**
- Easy to find and run
- Works from project root
- Simple for single developer

**Cons:**
- Root gets cluttered with many scripts
- Not ideal for large projects

---

### **Option 2: Scripts Directory (Recommended for Larger Projects)**

```
stock-recommendation-system/
├── scripts/
│   ├── start.sh          # Start all services
│   ├── stop.sh           # Stop all services
│   ├── restart.sh        # Restart services
│   ├── setup.sh          # Initial setup
│   ├── health-check.sh   # Check service health
│   └── dev.sh            # Development environment
├── backend/
├── frontend/
└── README.md
```

**Pros:**
- Organized and scalable
- Easy to add more scripts
- Professional structure

**Cons:**
- Need to navigate to scripts/ directory
- Slightly more setup

---

### **Option 3: Makefile (Best for Teams)**

```
stock-recommendation-system/
├── Makefile              # ← Single file for all commands
├── backend/
├── frontend/
└── README.md
```

**Pros:**
- Single file, very organized
- Standard for many projects
- Great for team collaboration
- Built-in file dependencies

**Cons:**
- Requires Makefile knowledge
- Tab indentation (can be tricky)

---

### **Option 4: Docker Compose (Best for Production)**

```
stock-recommendation-system/
├── docker-compose.yml    # ← Orchestrate all services
├── docker-compose.dev.yml
├── Dockerfile            # Backend
├── frontend/Dockerfile   # Frontend
├── backend/
└── frontend/
```

**Pros:**
- Containerized services
- Reproducible environments
- Easy deployment

**Cons:**
- Requires Docker knowledge
- Heavier setup

---

## 🚀 **Option 1: Simple Root-Level Scripts**

### **`start.sh` (Root Level)**

```bash
#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$PROJECT_ROOT/backend"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
LOG_DIR="$PROJECT_ROOT/logs"

# Create logs directory
mkdir -p "$LOG_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Stock Recommendation System - Startup${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to start a service
start_service() {
    local service=$1
    local port=$2
    local command=$3
    local log_file="$LOG_DIR/${service}.log"
    
    echo -e "${YELLOW}[${service}]${NC} Checking port ${port}..."
    
    if check_port $port; then
        echo -e "${RED}[${service}]${NC} Port ${port} already in use!"
        return 1
    fi
    
    echo -e "${YELLOW}[${service}]${NC} Starting service..."
    eval "$command" > "$log_file" 2>&1 &
    
    # Store PID
    echo $! >> "$LOG_DIR/pids.txt"
    
    # Wait and check if service started
    sleep 2
    
    if check_port $port; then
        echo -e "${GREEN}[${service}]${NC} Started successfully on port ${port}"
        echo -e "${GREEN}[${service}]${NC} Logs: $log_file"
        return 0
    else
        echo -e "${RED}[${service}]${NC} Failed to start"
        echo -e "${RED}[${service}]${NC} Check logs: $log_file"
        return 1
    fi
}

# Check if services are already running
echo -e "${YELLOW}Checking for existing services...${NC}\n"

# Clear old PID file
> "$LOG_DIR/pids.txt"

# 1. Start Ollama (LLM) - Port 11434
echo -e "${BLUE}1. LLM Service (Ollama)${NC}"
if check_port 11434; then
    echo -e "${GREEN}[Ollama]${NC} Already running on port 11434"
else
    echo -e "${YELLOW}[Ollama]${NC} Not running. Please start manually:"
    echo -e "${YELLOW}[Ollama]${NC}   ollama serve${NC}\n"
    echo -e "${YELLOW}[Ollama]${NC} Or install from: https://ollama.com"
fi

echo ""

# 2. Start Backend - Port 8000
echo -e "${BLUE}2. Backend Service (FastAPI)${NC}"
cd "$BACKEND_DIR" || exit

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}[Backend]${NC} Creating Python virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate 2>/dev/null || . venv/Scripts/activate 2>/dev/null

# Install/update dependencies
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}[Backend]${NC} Installing dependencies..."
    pip install -q -r requirements.txt
fi

# Start backend
start_service "Backend" "8000" "uvicorn app.main:app --reload"

echo ""

# 3. Start Frontend - Port 5173
echo -e "${BLUE}3. Frontend Service (React)${NC}"
cd "$FRONTEND_DIR" || exit

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}[Frontend]${NC} Installing dependencies..."
    npm install
fi

start_service "Frontend" "5173" "npm run dev"

echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Startup Complete!${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${GREEN}Services:${NC}"
echo -e "  Backend:  ${BLUE}http://localhost:8000${NC}"
echo -e "  Frontend: ${BLUE}http://localhost:5173${NC}"
echo -e "  API Docs: ${BLUE}http://localhost:8000/docs${NC}"

echo ""

echo -e "${YELLOW}To stop all services:${NC}"
echo -e "  ./stop.sh"

echo ""

echo -e "${YELLOW}Logs:${NC}"
echo -e "  $LOG_DIR/"

echo ""

# Keep script running
echo -e "${YELLOW}Press Ctrl+C to stop all services...${NC}\n"

# Wait for all background processes
wait
```

### **`stop.sh` (Root Level)**

```bash
#!/bin/bash

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Stopping All Services${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Function to stop a service by port
stop_service() {
    local service=$1
    local port=$2
    
    echo -e "${YELLOW}[${service}]${NC} Stopping on port ${port}..."
    
    # Find process on port and kill it
    if lsof -ti:$port | xargs kill -9 2>/dev/null; then
        echo -e "${GREEN}[${service}]${NC} Stopped successfully"
    else
        echo -e "${YELLOW}[${service}]${NC} Not running"
    fi
}

# Stop services
stop_service "Frontend" 5173
sleep 1

stop_service "Backend" 8000
sleep 1

echo ""
echo -e "${YELLOW}Note: Ollama must be stopped manually or with:${NC}"
echo -e "${YELLOW}  killall ollama${NC}\n"

echo -e "${GREEN}All services stopped!${NC}\n"
```

### **`restart.sh` (Root Level)**

```bash
#!/bin/bash

echo "Restarting all services..."
echo ""

./stop.sh

echo ""
echo "Waiting 2 seconds before restart..."
sleep 2

./start.sh
```

### **`health-check.sh` (Root Level)**

```bash
#!/bin/bash

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "Checking service health..."
echo ""

# Check Frontend
echo -n "Frontend (port 5173): "
if lsof -Pi :5173 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${RED}✗ Not running${NC}"
fi

# Check Backend
echo -n "Backend (port 8000): "
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Running${NC}"
    # Try API
    if curl -s http://localhost:8000/docs > /dev/null; then
        echo -e "         ${GREEN}✓ API responding${NC}"
    fi
else
    echo -e "${RED}✗ Not running${NC}"
fi

# Check Ollama
echo -n "Ollama (port 11434): "
if lsof -Pi :11434 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${RED}✗ Not running${NC}"
fi

echo ""
```

### **Make Scripts Executable**

```bash
chmod +x start.sh stop.sh restart.sh health-check.sh
```

---

## 🛠️ **Option 2: Makefile (Recommended for Teams)**

Create `Makefile` in project root:

```makefile
.PHONY: help start stop restart setup logs health clean

# Project directories
BACKEND_DIR := backend
FRONTEND_DIR := frontend
LOGS_DIR := logs

help:
	@echo "Stock Recommendation System - Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Initial setup (install deps, create venv)"
	@echo ""
	@echo "Running:"
	@echo "  make start          - Start all services"
	@echo "  make stop           - Stop all services"
	@echo "  make restart        - Restart all services"
	@echo "  make health         - Check service health"
	@echo ""
	@echo "Development:"
	@echo "  make backend        - Start only backend"
	@echo "  make frontend       - Start only frontend"
	@echo "  make logs           - Tail service logs"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove logs and cache"

setup:
	@echo "Setting up project..."
	@cd $(BACKEND_DIR) && python -m venv venv
	@cd $(BACKEND_DIR) && . venv/bin/activate && pip install -r requirements.txt
	@cd $(FRONTEND_DIR) && npm install
	@mkdir -p $(LOGS_DIR)
	@echo "Setup complete!"

start:
	@echo "Starting all services..."
	@mkdir -p $(LOGS_DIR)
	@cd $(BACKEND_DIR) && . venv/bin/activate && uvicorn app.main:app --reload > ../$(LOGS_DIR)/backend.log 2>&1 &
	@echo "Backend started (port 8000)"
	@sleep 1
	@cd $(FRONTEND_DIR) && npm run dev > ../$(LOGS_DIR)/frontend.log 2>&1 &
	@echo "Frontend started (port 5173)"
	@echo ""
	@echo "Services running:"
	@echo "  Frontend:  http://localhost:5173"
	@echo "  Backend:   http://localhost:8000"
	@echo "  API Docs:  http://localhost:8000/docs"

stop:
	@echo "Stopping all services..."
	@lsof -ti:5173 | xargs kill -9 2>/dev/null || true
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@echo "Services stopped!"

restart: stop
	@sleep 1
	@$(MAKE) start

backend:
	@cd $(BACKEND_DIR) && . venv/bin/activate && uvicorn app.main:app --reload

frontend:
	@cd $(FRONTEND_DIR) && npm run dev

logs:
	@echo "=== Backend Logs ==="
	@tail -f $(LOGS_DIR)/backend.log &
	@echo "=== Frontend Logs ==="
	@tail -f $(LOGS_DIR)/frontend.log &
	@wait

health:
	@echo "Checking service health..."
	@echo -n "Frontend: "
	@lsof -Pi :5173 -sTCP:LISTEN -t >/dev/null 2>&1 && echo "✓ Running" || echo "✗ Not running"
	@echo -n "Backend:  "
	@lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 && echo "✓ Running" || echo "✗ Not running"

clean:
	@echo "Cleaning up..."
	@rm -rf $(LOGS_DIR)/*
	@cd $(FRONTEND_DIR) && npm cache clean --force
	@echo "Cleanup complete!"
```

**Usage:**

```bash
make help        # See all commands
make setup       # First time setup
make start       # Start all services
make stop        # Stop services
make health      # Check status
make logs        # View logs
```

---

## 🐳 **Option 3: Docker Compose**

Create `docker-compose.yml` in project root:

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - LLM_PROVIDER=ollama
      - OLLAMA_BASE_URL=http://ollama:11434
    volumes:
      - ./backend/data:/app/data
    command: uvicorn app.main:app --host 0.0.0.0 --reload

  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    environment:
      - VITE_API_URL=http://localhost:8000
    volumes:
      - ./frontend/src:/app/src

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    command: serve

volumes:
  ollama_data:
```

**Usage:**

```bash
docker-compose up          # Start all services
docker-compose down        # Stop services
docker-compose logs -f     # View logs
```

---

## 📊 **Comparison Table**

| Method | Ease | Scalability | Team-Friendly | Production |
|--------|------|-------------|---------------|-----------|
| Root Bash Scripts | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐ |
| Scripts Directory | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| Makefile | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Docker Compose | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🎯 **My Recommendation**

### **For Solo Developer:**
Use **Option 1: Root-level Bash Scripts** (`start.sh`, `stop.sh`)
- Simplest
- Easy to understand
- Quick to get running

### **For Small Team:**
Use **Option 2: Makefile**
- Professional
- Easy for others to use (`make start`)
- Scalable

### **For Production/DevOps:**
Use **Option 3: Docker Compose**
- Reproducible
- Works on any machine
- Easy deployment

---

## 📝 **Quick Setup Instructions**

### **Using Root-Level Scripts (Easiest)**

```bash
# 1. Copy scripts to project root
cp start.sh stop.sh restart.sh health-check.sh \
  /path/to/stock-recommendation-system/

# 2. Make executable
chmod +x /path/to/stock-recommendation-system/*.sh

# 3. Start everything
cd /path/to/stock-recommendation-system
./start.sh

# 4. View logs
tail -f logs/backend.log
tail -f logs/frontend.log

# 5. Stop everything
./stop.sh
```

### **Using Makefile**

```bash
# 1. Copy Makefile to project root
cp Makefile /path/to/stock-recommendation-system/

# 2. Initial setup
cd /path/to/stock-recommendation-system
make setup

# 3. Start everything
make start

# 4. Check health
make health

# 5. Stop everything
make stop
```

---

## 🏆 **Best Practice Summary**

1. **Place scripts in a dedicated location**
   - Not scattered in different directories
   - Easy to find and maintain

2. **Make scripts executable**
   ```bash
   chmod +x script.sh
   ```

3. **Add proper error handling**
   - Check if services are already running
   - Validate dependencies exist
   - Provide helpful error messages

4. **Document everything**
   - Add comments to scripts
   - Create README for running services
   - Include troubleshooting section

5. **Version control**
   - Commit scripts to Git
   - Makes onboarding new developers easy
   - Reproducible setup for everyone

---

## 📋 **Starter README Section**

Add to `README.md`:

```markdown
## Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+
- Ollama (for LLM)

### Quick Start

#### Option 1: Using Bash Scripts
```bash
./start.sh
```

#### Option 2: Using Make
```bash
make start
```

#### Option 3: Docker Compose
```bash
docker-compose up
```

### Access Services
- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Stopping Services

#### Bash
```bash
./stop.sh
```

#### Make
```bash
make stop
```

#### Docker
```bash
docker-compose down
```
```

---

**My final recommendation: Start with Option 1 (root-level scripts) for simplicity, then upgrade to Makefile when working with a team!**

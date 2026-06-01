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

# 1. Check Ollama (LLM) - Port 11434
echo -e "${BLUE}1. LLM Service (Ollama)${NC}"
if check_port 11434; then
    echo -e "${GREEN}[Ollama]${NC} Already running on port 11434\n"
else
    echo -e "${YELLOW}[Ollama]${NC} Not running. Please start manually:"
    echo -e "${YELLOW}[Ollama]${NC}   ollama serve${NC}"
    echo -e "${YELLOW}[Ollama]${NC}   Or download from: https://ollama.com\n"
fi

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

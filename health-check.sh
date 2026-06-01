#!/bin/bash

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Service Health Check${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check Frontend
echo -n "Frontend (port 5173):     "
if lsof -Pi :5173 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${RED}✗ Not running${NC}"
fi

# Check Backend
echo -n "Backend (port 8000):      "
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Running${NC}"
    # Try API
    if curl -s http://localhost:8000/docs > /dev/null 2>&1; then
        echo -e "                          ${GREEN}✓ API responding${NC}"
    else
        echo -e "                          ${YELLOW}⚠ API not responding${NC}"
    fi
else
    echo -e "${RED}✗ Not running${NC}"
fi

# Check Ollama
echo -n "Ollama (port 11434):      "
if lsof -Pi :11434 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Running${NC}"
else
    echo -e "${RED}✗ Not running${NC}"
fi

echo ""
echo -e "${BLUE}To start all services:${NC} ./start.sh"
echo -e "${BLUE}To stop all services:${NC} ./stop.sh"
echo ""

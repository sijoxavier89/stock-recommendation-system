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

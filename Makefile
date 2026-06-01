.PHONY: help start stop restart setup logs health clean backend frontend

# Project directories
BACKEND_DIR := backend
FRONTEND_DIR := frontend
LOGS_DIR := logs

help:
	@echo "Stock Recommendation System - Available Commands"
	@echo ""
	@echo "Setup (first time only):"
	@echo "  make setup          - Install dependencies for backend and frontend"
	@echo ""
	@echo "Running Services:"
	@echo "  make start          - Start all services (Backend + Frontend)"
	@echo "  make stop           - Stop all services"
	@echo "  make restart        - Restart all services"
	@echo "  make health         - Check service health status"
	@echo ""
	@echo "Development:"
	@echo "  make backend        - Start only backend (port 8000)"
	@echo "  make frontend       - Start only frontend (port 5173)"
	@echo "  make logs           - Tail logs from all services"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean          - Remove logs and caches"
	@echo ""

setup:
	@echo "Setting up project..."
	@cd $(BACKEND_DIR) && python -m venv venv
	@cd $(BACKEND_DIR) && . venv/bin/activate && pip install -r requirements.txt || true
	@cd $(FRONTEND_DIR) && npm install || true
	@mkdir -p $(LOGS_DIR)
	@echo "✓ Setup complete!"
	@echo ""
	@echo "Next step: make start"

start:
	@echo "Starting all services..."
	@mkdir -p $(LOGS_DIR)
	@cd $(BACKEND_DIR) && . venv/bin/activate && uvicorn app.main:app --reload > ../$(LOGS_DIR)/backend.log 2>&1 &
	@echo "✓ Backend started (port 8000)"
	@sleep 1
	@cd $(FRONTEND_DIR) && npm run dev > ../$(LOGS_DIR)/frontend.log 2>&1 &
	@echo "✓ Frontend started (port 5173)"
	@echo ""
	@echo "Services are running:"
	@echo "  Frontend:  http://localhost:5173"
	@echo "  Backend:   http://localhost:8000"
	@echo "  API Docs:  http://localhost:8000/docs"
	@echo ""
	@echo "View logs with: make logs"

stop:
	@echo "Stopping all services..."
	@lsof -ti:5173 | xargs kill -9 2>/dev/null || echo "Frontend not running"
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || echo "Backend not running"
	@echo "✓ Services stopped!"

restart: stop
	@sleep 1
	@$(MAKE) start

backend:
	@cd $(BACKEND_DIR) && . venv/bin/activate && uvicorn app.main:app --reload

frontend:
	@cd $(FRONTEND_DIR) && npm run dev

logs:
	@mkdir -p $(LOGS_DIR)
	@echo "Tailing logs (Ctrl+C to stop)..."
	@echo ""
	@tail -f $(LOGS_DIR)/backend.log $(LOGS_DIR)/frontend.log 2>/dev/null || echo "No logs found. Run 'make start' first."

health:
	@echo "Checking service health..."
	@echo ""
	@echo -n "Frontend (port 5173): "
	@lsof -Pi :5173 -sTCP:LISTEN -t >/dev/null 2>&1 && echo "✓ Running" || echo "✗ Not running"
	@echo -n "Backend (port 8000):  "
	@lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1 && echo "✓ Running" || echo "✗ Not running"
	@echo -n "Ollama (port 11434):  "
	@lsof -Pi :11434 -sTCP:LISTEN -t >/dev/null 2>&1 && echo "✓ Running" || echo "✗ Not running"
	@echo ""

clean:
	@echo "Cleaning up..."
	@rm -rf $(LOGS_DIR)/*
	@cd $(FRONTEND_DIR) && npm cache clean --force 2>/dev/null || true
	@echo "✓ Cleanup complete!"

.DEFAULT_GOAL := help

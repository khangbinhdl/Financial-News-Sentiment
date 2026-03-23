.PHONY: help install install-deps run-api run-ui run dev clean format lint test

install: install-deps ## Install all dependencies

install-deps: ## Install required Python packages
	pip install -r requirements.txt

run-api: ## Run FastAPI server
	@echo "Starting FastAPI server on http://127.0.0.1:8000"
	python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

run-ui: ## Run Streamlit app
	@echo "Starting Streamlit app on http://localhost:8501"
	streamlit run ui/app.py --server.port=8501 --server.address=localhost

dev: ## Run both API and UI servers in parallel
	@echo "Starting both servers..."
	@echo "API: http://127.0.0.1:8000"
	@echo "UI: http://localhost:8501"
	@make -j 2 run-api run-ui

run: dev ## Alias for dev (run both servers)


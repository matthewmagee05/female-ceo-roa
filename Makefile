# Makefile for Female CEO ROA Analysis

.PHONY: help install install-test test test-unit test-integration test-cov clean run run-refactored lint format

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install main dependencies
	pip install -r requirements.txt

install-test: ## Install test dependencies
	pip install -r requirements-test.txt

test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/ -v -m "not integration"

test-integration: ## Run integration tests only
	pytest tests/ -v -m "integration"

test-cov: ## Run tests with coverage
	pytest tests/ --cov=src --cov-report=html --cov-report=term

clean: ## Clean up generated files
	rm -rf out/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run: ## Run the analysis script
	.venv/bin/python female_ceo_roa.py

lint: ## Run linting (if you have flake8 or similar installed)
	@echo "Linting not configured - add flake8, black, or similar tools"

format: ## Format code (if you have black installed)
	@echo "Formatting not configured - add black or similar tools"

setup: install install-test ## Setup the project (install all dependencies)
	@echo "Project setup complete!"

ci: test-cov ## Run CI pipeline (tests with coverage)
	@echo "CI pipeline complete!"

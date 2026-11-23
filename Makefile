.PHONY: install lint format typecheck test check demo dashboard docker clean

install:  ## install package with dev extras
	pip install -e ".[dev]"

lint:
	ruff check src tests examples
	ruff format --check src tests examples

format:
	ruff check --fix src tests examples
	ruff format src tests examples

typecheck:
	mypy

test:
	pytest --cov --cov-report=term-missing

check: lint typecheck test  ## everything CI runs

demo:  ## offline red-team demo: naive vs guarded toy bot, no API keys needed
	evalharness run examples/offline_demo.yaml --label naive
	APP=guarded_bot evalharness run examples/offline_demo.yaml --label guarded
	evalharness compare naive__prompt_injection guarded__prompt_injection
	evalharness gate guarded__prompt_injection --baseline naive__prompt_injection

dashboard:
	streamlit run dashboard/app.py

docker:
	docker build -t evalharness:latest .

clean:
	rm -rf .evalharness .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build

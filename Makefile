.PHONY: install install-dev test unit theory smoke isolation lint format typecheck smoke-gate clean demo-attack demo-rce

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

unit:
	pytest tests/unit -v

theory:
	pytest tests/theory -v

property:
	pytest tests/property -v

smoke:
	pytest tests/smoke -v

isolation:
	pytest tests/isolation -v

smoke-gate:
	python scripts/smoke_test.py

lint:
	ruff check src/ scripts/ tests/

format:
	ruff format src/ scripts/ tests/

typecheck:
	mypy src/safelie

demo-clean:
	python scripts/train.py --config configs/experiment/local_demo_clean.yaml

demo-attack:
	python scripts/train.py --config configs/experiment/local_demo_attack.yaml

demo-rce:
	python scripts/train.py --config configs/experiment/local_demo_rce.yaml

demo-benign:
	python scripts/train.py --config configs/experiment/local_demo_benign.yaml

audit-m5:
	python scripts/audit_sources.py --preset m5_two_agent --assumed-f 2

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache

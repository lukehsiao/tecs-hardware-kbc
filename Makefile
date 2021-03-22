dev:
	poetry install
	poetry run pre-commit install

check:
	isort -c hack/
	isort -c scripts/
	isort -c bin/
	black --check hack/
	black --check scripts/
	black --pyi --check bin/*
	flake8 hack/
	flake8 scripts/
	flake8 bin/

.PHONY: dev check

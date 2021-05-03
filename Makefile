dev:
	poetry install
	poetry run pre-commit install

check:
	isort -c hack/
	isort -c scripts/
	black --check hack/
	black --check scripts/
	flake8 hack/
	flake8 scripts/

.PHONY: dev check

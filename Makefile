dev:
	pip install -r requirements.txt
	pip install -e . --no-use-pep517
	pre-commit install

check:
	isort -rc -c hack/
	isort -rc -c scripts/
	isort -rc -c bin/
	black --check hack/
	black --check scripts/
	black --pyi --check bin/*
	flake8 hack/
	flake8 scripts/
	flake8 bin/

clean:
	pip uninstall -y hack 
	rm -rf hack.egg-info pip-wheel-metadata

.PHONY: dev clean check

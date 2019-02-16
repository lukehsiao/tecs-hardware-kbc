dev:
	pip install -r requirements.txt
	pip install -e .
	pre-commit install

clean:
	pip uninstall -y hack 
	rm -rf hack.egg-info

.PHONY: dev clean

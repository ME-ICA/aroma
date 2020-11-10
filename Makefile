.PHONY: all lint

all_tests: lint unittest integration

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  lint			to run flake8 on all Python files"
	@echo "  unittest		to run unit tests on aroma"
	@echo "  integration	to run the integration test set on aroma"
	@echo "  all_tests		to run 'lint', 'unittest', and 'integration'"

lint:
	@flake8 aroma

unittest:
	@py.test --skipintegration --cov-append --cov-report term-missing --cov=aroma aroma/

integration:
	@pip install -e ".[test]"
	@py.test --log-cli-level=INFO --cov-append --cov-report term-missing --cov=aroma -k test_integration aroma/tests/test_integration.py
export PYTHONPATH := .:transformer:$(PYTHONPATH)

PACKAGE_TRAINING=training
PACKAGE_INFERENCE=inference
MODULE_SCRIPTS=scripts
MODULES=$(MODULE_SCRIPTS) $(MODULE_TRANSFORMER) $(PACKAGE_TRAINING) $(PACKAGE_INFERENCE)
DOC_MODULES=$(MODULE_TRANSFORMER) $(PACKAGE_TRAINING) $(PACKAGE_INFERENCE)
UNIT_TESTS=tests/unit

ifeq ($(CICD), TRUE)
	PYTEST_FLAGS = -p no:warnings
else
	POETRY_ARG = poetry run
	PYTEST_FLAGS = -v -s
endif

# Check README.md to generate this file
SSH_FILE := $${HOME}/.ssh/keys/wk_github

all: static-tests doc-tests coverage

.PHONY: all

style:
	###### Running style analysis ######
	$(POETRY_ARG) flake8 $(MODULES)
	$(POETRY_ARG) flake8 $(UNIT_TESTS)

typecheck:
	###### Running static type analysis ######
	$(POETRY_ARG) mypy $(MODULE_SCRIPTS)
	$(POETRY_ARG) mypy $(PACKAGE_INFERENCE)
	$(POETRY_ARG) mypy $(PACKAGE_TRAINING)

doccheck:
	###### Running documentation analysis ######
	$(POETRY_ARG) pydocstyle $(MODULES)

static-tests: style typecheck doccheck

unit-tests:
	###### Running unit tests ######
	$(POETRY_ARG) pytest $(PYTEST_FLAGS) $(UNIT_TESTS)

doc-tests:
  	###### Running unit tests ######
	$(POETRY_ARG) pytest --doctest-modules $(PYTEST_FLAGS) $(DOC_MODULES)

coverage:
	###### Running coverage analysis ######
	$(POETRY_ARG) pytest $(PYTEST_FLAGS) tests/unit/$(PACKAGE_INFERENCE) --cov $(PACKAGE_INFERENCE) --cov-report term-missing
	$(POETRY_ARG) pytest $(PYTEST_FLAGS) tests/unit/$(PACKAGE_TRAINING) --cov $(PACKAGE_TRAINING) --cov-report term-missing

coverage-html:
	###### Running coverage analysis with html export ######
	$(POETRY_ARG) pytest $(PYTEST_FLAGS) tests/unit/$(PACKAGE_INFERENCE) --cov $(PACKAGE_INFERENCE) --cov-report html
	$(POETRY_ARG) pytest $(PYTEST_FLAGS) tests/unit/$(PACKAGE_TRAINING) --cov $(PACKAGE_TRAINING) --cov-report html
	open htmlcov/index.html

coverage-xml:
	###### Running coverage analysis with JUnit xml export ######
	$(POETRY_ARG) pytest -v --cov $(PACKAGE_INFERENCE) --junitxml=junit.xml
	$(POETRY_ARG) pytest -v --cov $(PACKAGE_TRAINING) --junitxml=junit.xml

## Test using a Docker Container
docker-tests:
	###### Docker build and run tests ######
	docker build --rm --tag nlp-training:test --target tester .
	docker run --name nlp-training --rm nlp-training:test

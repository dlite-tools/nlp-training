# Basic definitions
export PYTHONPATH := .:$(PYTHONPATH)

PACKAGE_TRAINING=training
PACKAGE_INFERENCE=inference
MODULE_SCRIPTS=scripts

MODULES=$(MODULE_SCRIPTS) $(PACKAGE_TRAINING) $(PACKAGE_INFERENCE)
DOC_MODULES=$(PACKAGE_TRAINING) $(PACKAGE_INFERENCE)

UNIT_TESTS=tests/unit

# Run test out of Poetry environment
ifeq ($(CICD), TRUE)
	PYTEST_FLAGS = -p no:warnings
else
	POETRY_ARG = poetry run
	PYTEST_FLAGS = -v -s
endif

## Scripts to run tests

static-tests:
	###### Running style analysis ######
	$(POETRY_ARG) flake8 $(MODULES)
	# $(POETRY_ARG) flake8 $(UNIT_TESTS)

	###### Running static type analysis ######
	$(POETRY_ARG) mypy $(MODULE_SCRIPTS)
	$(POETRY_ARG) mypy $(PACKAGE_INFERENCE)
	$(POETRY_ARG) mypy $(PACKAGE_TRAINING)

	###### Running documentation analysis ######
	$(POETRY_ARG) pydocstyle $(MODULES)

coverage:
	###### Running unit tests and coverage analysis ######
	$(POETRY_ARG) pytest $(PYTEST_FLAGS) tests/unit/$(PACKAGE_INFERENCE) --cov $(PACKAGE_INFERENCE) --cov-report term-missing
	$(POETRY_ARG) pytest $(PYTEST_FLAGS) tests/unit/$(PACKAGE_TRAINING) --cov $(PACKAGE_TRAINING) --cov-report term-missing

tests: static-tests coverage

## Scripts to run training

train:
	$(POETRY_ARG) python scripts/train.py

## Scripts to run Docker commands
docker-tests:
	###### Docker build and run tests ######
	docker build --rm --tag nlp-training:test .
	docker run --name nlp-training-test --rm nlp-training:test

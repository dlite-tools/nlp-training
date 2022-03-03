######### BASE Image #########
FROM python:3.8-buster AS base

ENV PATH="/poetry/bin:$PATH" \
    POETRY_VERSION="1.1.12" \
    SERVICE_HOME="/nlp"

WORKDIR ${SERVICE_HOME}

COPY poetry.lock pyproject.toml ${SERVICE_HOME}/

RUN libDeps='ffmpeg libsm6 libxext6' && \
    apt-get update -y -qq > /dev/null && \
    apt-get -y -qq install $libDeps --no-install-recommends > /dev/null && \
    curl -s https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py --output /install-poetry.py && \
    POETRY_HOME=/poetry python /install-poetry.py && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-root --no-dev && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/* /root/.ssh

######### PRODUCTION Image #########
FROM python:3.8-buster AS production

ENV SERVICE_HOME="/nlp"

WORKDIR ${SERVICE_HOME}

ENV PATH="$SERVICE_HOME/.venv/bin:$PATH" \
    PYTHONPATH="$SERVICE_HOME:$SERVICE_HOME/.venv/lib/python3.8/site-packages/"

COPY --from=base ${SERVICE_HOME}/.venv ${SERVICE_HOME}/.venv

COPY ./inference ${SERVICE_HOME}/inference

######### TEST Image #########
FROM production AS tester

ENV CICD=TRUE \
    PATH="/poetry/bin:$PATH" \
    POETRY_VERSION="1.1.12"

COPY poetry.lock pyproject.toml ./

COPY --from=base /install-poetry.py /install-poetry.py

RUN libDeps='build-essential curl ssh ffmpeg libsm6 libxext6' && \
    apt-get update -y -qq > /dev/null && \
    apt-get -y -qq install $libDeps --no-install-recommends > /dev/null && \
    touch /root/.ssh/known_hosts && \
    curl -s https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py --output /install-poetry.py && \
    POETRY_HOME=/poetry python /install-poetry.py && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-root --extras training && \
    # TODO: Remove this line in a future release of Pytorch
    pip install setuptools==59.5.0 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/* /root/.ssh

COPY . ${SERVICE_HOME}

CMD make all

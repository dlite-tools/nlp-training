FROM python:3.9-buster

ARG homedir=/home/nlp
ARG poetry_version=1.1.12

ENV PATH="/poetry/bin:$homedir/.venv/bin:$PATH" \
    POETRY_VERSION=$poetry_version \
    PYTHONPATH="$homedir:$homedir/.venv/lib/python3.9/site-packages/" \
    # Runs outside Poetry virtual environment
    CICD=TRUE

RUN mkdir -p $homedir

WORKDIR $homedir

COPY poetry.lock pyproject.toml $homedir/

RUN libDeps='ffmpeg libsm6 libxext6' && \
    apt-get update -y -qq > /dev/null && \
    apt-get -y -qq install $libDeps --no-install-recommends > /dev/null && \
    curl -s https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py --output /install-poetry.py && \
    POETRY_HOME=/poetry python /install-poetry.py && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-root --extras training && \
    # TODO: Remove this line in a future release of Pytorch
    pip install setuptools==59.5.0 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/* /tmp/* /var/tmp/*

COPY . $homedir/

CMD make tests

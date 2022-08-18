# syntax=docker/dockerfile:1
ARG PARENT_IMAGE
ARG USE_GPU
FROM $PARENT_IMAGE

RUN apt-get -y update \
    && apt-get -y install \
    sysstat \
    ffmpeg \
    freeglut3-dev \
    swig \
    xvfb \
    libxrandr2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV CODE_DIR /root/code
ENV VENV /root/venv
COPY requirements.txt /tmp/


RUN \
    mkdir -p ${CODE_DIR}/rl_zoo && \
    pip uninstall -y stable-baselines3 && \
    pip install -r /tmp/requirements.txt && \
    pip install pip install highway-env==1.5.0 && \
    rm -rf $HOME/.cache/pip

ENV PATH=$VENV/bin:$PATH

COPY utils/ ${CODE_DIR}/rl_zoo/utils/
COPY train.py ${CODE_DIR}/rl_zoo/train.py
COPY hyperparams/ ${CODE_DIR}/rl_zoo/hyperparams/
COPY scripts/ ${CODE_DIR}/rl_zoo/scripts/
COPY run_multi_train.py ${CODE_DIR}/rl_zoo/run_multi_train.py

CMD /bin/bash

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# from: https://github.com/sgugger/torchdynamo-tests
# build with:
# `docker build -f Dockerfile --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t container-need4speed .`

# run with `docker run --gpus device=4 --rm -it -v $(pwd)/scripts:/workspace container-need4speed:latest python run.py --config_path all`

ENV PATH="/home/user/miniconda3/bin:${PATH}"
ARG PATH="/home/user/miniconda3/bin:${PATH}"

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    git && \
    apt-get clean

USER user
WORKDIR /home/user

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir .conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda init bash

RUN pip install -r requirements.txt

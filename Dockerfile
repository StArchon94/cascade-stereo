FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
LABEL maintainer=slin@ttic.edu
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago
RUN apt update && apt install -y \
    python3-opencv \
    tmux \
    unzip \
    wget \
    && apt full-upgrade -y && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    numpy \
    opencv-python \
    pillow \
    tensorboardX
RUN wget https://github.com/NVIDIA/apex/archive/refs/heads/master.zip \
    && unzip master.zip \
    && cd apex-master/ \
    && pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . \
    && rm -r ../apex-master/
WORKDIR /root

FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
LABEL maintainer=slin@ttic.edu
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Chicago
RUN apt update && apt install -y \
    cmake \
    libopencv-dev \
    mesa-common-dev \
    tmux \
    unzip \
    wget \
    && apt full-upgrade -y && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir \
    numpy \
    opencv-python \
    pillow \
    plyfile \
    tensorboardX
RUN wget https://github.com/NVIDIA/apex/archive/refs/heads/master.zip \
    && unzip master.zip && rm master.zip \
    && cd apex-master/ \
    && pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . \
    && rm -r ../apex-master/
WORKDIR /root
RUN wget https://github.com/YoYo000/fusibile/archive/refs/heads/master.zip \
    && unzip master.zip && rm master.zip \
    && mv fusibile-master fusibile && cd fusibile \
    && cmake . \
    && make -j

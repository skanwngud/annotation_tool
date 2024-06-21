FROM ubuntu:20.04
LABEL version="front:init"


RUN apt-get update -y && apt-get install -y sudo
RUN adduser --disabled-password --gecos "" user \
    && echo "user:user" | chpasswd \
    && adduser user sudo \
    && echo "user ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN apt-get update -y && apt-get install -y \
    vim \
    python3 \
    python3-pip \
    net-tools

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update -y && apt-get install -y tzdata
RUN apt-get update -y && apt-get install -y \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6

RUN pip install \
    opencv-python \
    opencv-contrib-python \
    fastapi \
    uvicorn \
    requests \
    loguru \
    pillow

WORKDIR /home/user/source

RUN mkdir templates

COPY main.py /home/user/source
# COPY utils.py /home/user/source
COPY classes.json /home/user/source
COPY templates/select_options.html /home/user/source/templates
COPY templates/go_back.html /home/user/source/templates
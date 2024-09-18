FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y; \
    apt-get install python3.8 -y;   \
    apt-get install python3-pip -y; \
    python3.8 -m pip install --upgrade pip; \
    apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y; \
    apt-get install python3.8-dev -y;   \
    apt-get install -y git; \
    python3.8 -m pip install wheel; \
    pip install --upgrade pip;  \
    pip install torch==1.13.0+cu117 \
    torchvision==0.14.0+cu117 \
    torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

WORKDIR /app-src
COPY requirements.txt /app-src/

RUN python3.8 -m pip install -r requirements.txt

COPY . /app-src/
RUN git config --global --add safe.directory /app-src/face-anti-spoofing-training
# CMD ["bash", "./run.sh"]


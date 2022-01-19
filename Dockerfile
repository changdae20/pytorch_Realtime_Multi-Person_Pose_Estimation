FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-devel

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && apt-get upgrade -y && \
    apt-get -y install git libgl1-mesa-glx libglib2.0-0 libomp-dev wget software-properties-common

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt


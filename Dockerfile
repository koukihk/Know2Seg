FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# FROM docker.mirrors.ustc.edu.cn/nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3-pip \
    python3.8-dev \
    build-essential \
    libopenblas-dev \
    tzdata \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN python3 -m pip install --upgrade "pip==23.3.1" "setuptools==58.0.4" "wheel==0.37.1" -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install "pytz==2022.7" "numpy==1.23.5" --prefer-binary -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install -r requirements.txt \
    --use-deprecated=legacy-resolver \
    --no-build-isolation \
    --prefer-binary \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

VOLUME /dataset

WORKDIR /app

ENTRYPOINT ["python3.8"]
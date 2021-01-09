FROM tensorflow/tensorflow:latest-gpu

WORKDIR /workspace

COPY . /workspace

RUN pip install --no-cache-dir -r requirements.txt
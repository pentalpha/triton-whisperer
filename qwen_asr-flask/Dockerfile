FROM nvcr.io/nvidia/tritonserver:24.08-py3

RUN pip install transformers torch numpy requests librosa tritonclient[http] \
    && pip cache purge
RUN pip install datasets[audio] accelerate \
    && pip cache purge
RUN apt update && apt install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torchcodec \
    && pip cache purge
RUN pip install torchvision
RUN python3 -c "import torch; import torchvision; print(torchvision.ops.nms(torch.zeros(1,4), torch.ones(1), 0.5))"

RUN pip install -U qwen-asr  --ignore-installed blinker --break-system-packages
RUN python3 -c "import torch; import torchvision; print(torchvision.ops.nms(torch.zeros(1,4), torch.ones(1), 0.5))"

RUN pip flask psutil --break-system-packages

COPY server.py /server.py
EXPOSE 8000

CMD ["python3 /server.py"]


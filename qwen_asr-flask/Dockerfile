FROM nvcr.io/nvidia/tritonserver:25.10-pyt-python-py3
RUN python3 -c "import torch; import torchvision; print(torchvision.ops.nms(torch.zeros(1,4), torch.ones(1), 0.5))"
RUN pip install -U qwen-asr --ignore-installed blinker --break-system-packages
RUN python3 -c "import torch; import torchvision; print(torchvision.ops.nms(torch.zeros(1,4), torch.ones(1), 0.5))"

RUN apt update && apt -y install libcusparselt0 libcusparselt-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 1. Install base packages
RUN pip install numpy requests librosa --break-system-packages

# 2. Install torchvision WITHOUT dependencies so it cannot overwrite Triton's PyTorch
RUN pip install torchvision --no-deps --break-system-packages
RUN python3 -c "import torch; import torchvision; print(torchvision.ops.nms(torch.zeros(1,4), torch.ones(1), 0.5))"

# 3. Install the rest of stack
RUN pip install transformers datasets[audio] accelerate torchcodec --break-system-packages
RUN python3 -c "import torch; import torchvision; print(torchvision.ops.nms(torch.zeros(1,4), torch.ones(1), 0.5))"

RUN pip flask psutil --break-system-packages

COPY server.py /server.py
EXPOSE 8000

CMD ["python3 /server.py"]


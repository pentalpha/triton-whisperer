FROM nvcr.io/nvidia/tritonserver:25.10-pyt-python-py3
RUN python3 -c "import torch; import torchvision; print(torchvision.ops.nms(torch.zeros(1,4), torch.ones(1), 0.5))"
RUN pip install qwen-asr[vllm] --ignore-installed blinker --break-system-packages
RUN python3 -c "import torch; import torchvision; print(torchvision.ops.nms(torch.zeros(1,4), torch.ones(1), 0.5))"

RUN apt update && apt -y install libcusparselt0 libcusparselt-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 1. Install base packages
RUN pip install numpy requests librosa tritonclient[http] --break-system-packages

# 2. Install torchvision WITHOUT dependencies so it cannot overwrite Triton's PyTorch
RUN pip install torchvision --no-deps --break-system-packages
RUN python3 -c "import torch; import torchvision; print(torchvision.ops.nms(torch.zeros(1,4), torch.ones(1), 0.5))"

# 3. Install the rest of stack
RUN pip install transformers datasets[audio] accelerate torchcodec --break-system-packages
RUN python3 -c "import torch; import torchvision; print(torchvision.ops.nms(torch.zeros(1,4), torch.ones(1), 0.5))"


COPY ./model_repository /models
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

CMD ["tritonserver", "--log-verbose", "1", "--model-repository=/models", "--load-model=qwen3_asr_1.7b", "--model-control-mode=explicit"]
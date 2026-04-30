cd /home/pita/triton-whisperer/qwen_asr  \
 && sudo docker build -t triton-qwen_asr ./ \
 && sudo docker run --runtime=nvidia --gpus all --shm-size 1G --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v /home/pita/hf_models:/root/.cache/huggingface triton-qwen_asr tritonserver --log-verbose 1 --model-repository=/models --load-model=qwen3_asr_1.7b --model-control-mode=explicit
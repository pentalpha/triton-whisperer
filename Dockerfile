FROM nvcr.io/nvidia/tritonserver:25.09-py3
ENV CUDA_VISIBLE_DEVICES=""

COPY model_repository/whisper_cpu/requirements.txt /models/whisper_cpu/requirements.txt
RUN pip install -r /models/whisper_cpu/requirements.txt \
    && rm /models/whisper_cpu/requirements.txt

COPY ./model_repository /models
EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

CMD ["tritonserver", "--log-verbose", "1", "--model-repository=/models"]
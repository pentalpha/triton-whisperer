# Triton Whisperer

Repository which runs Whisper model inside a NVIDIA Triton container, reads .wav files in a mounted directory and creates .txt files with the transcriptions. By default, the triton container runs on CPU.

## Build Custom Container

```bash
sudo docker build -t triton-whisper ./
docker run -p8000:8000 -p8001:8001 -p8002:8002 -v ${PWD}/hf_models:/root/.cache/huggingface triton-whisper
```


### Model Repository Structure

Ensure your model repository has the following structure:

```
model_repository/
  └── whisper_cpu/
      ├── config.pbtxt
      ├── requirements.txt
      └── 1/
          └── model.py
```

- `model_repository/whisper_cpu/config.pbtxt`: Configuration for the Triton model.
- `model_repository/whisper_cpu/requirements.txt`: Python dependencies for the model.
- `model_repository/whisper_cpu/1/model.py`: The Python backend script implementing `TritonPythonModel`.

## Using the Model (Client Script)

This repository provides a `client.py` script to send `.wav` files for transcription to the running Triton server.

#### Install Client Dependencies (on Host Machine)
Before running the client script, ensure you have the necessary dependencies installed on your host machine (outside the Docker container):

```bash
pip install librosa tritonclient[http]
```

#### Run the Client Script
To use the client, provide the directory containing your `.wav` audio files:

```bash
python3 client.py /path/to/your/audio_directory
```

The script will:
1.  Find all `.wav` files in the specified directory.
2.  Preprocess each audio file to match the model's input requirements.
3.  Send the preprocessed audio to the Triton server.
4.  Receive the transcription from the server.
5.  Print the transcription to the console.
6.  Save the transcription to a `.txt` file with the same name as the original `.wav` file in the same directory (e.g., `audio.wav` -> `audio.txt`).
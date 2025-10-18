import argparse
import glob
import os
from time import time
import librosa
import numpy as np
import tritonclient.http as http_client

MODEL_NAME = "whisper_cpu"
TRITON_SERVER_URL = "localhost:8000"

def preprocess_audio(audio_path):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz
    return audio.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Triton Whisper Client for WAV files.")
    parser.add_argument("directory", type=str, help="Directory containing .wav files.")
    args = parser.parse_args()

    wav_files = glob.glob(os.path.join(args.directory, "*.wav"))
    timeout_secs = 240
    if not wav_files:
        print(f"No .wav files found in {args.directory}")
        return

    print(f"Found {len(wav_files)} .wav files. Sending to Triton server...")

    client = http_client.InferenceServerClient(url=TRITON_SERVER_URL,
        connection_timeout=timeout_secs,
        network_timeout=timeout_secs)

    for wav_file in wav_files:
        print(f"Processing {wav_file}...")
        try:
            # Preprocess audio
            audio_input_data = preprocess_audio(wav_file)
            
            # Create input tensor
            inputs = [
                http_client.InferInput(
                    "INPUT_0", audio_input_data.shape, "FP32"
                )
            ]
            inputs[0].set_data_from_numpy(audio_input_data)

            # Send request
            infer_start_time = time()
            results = client.infer(model_name=MODEL_NAME, inputs=inputs, timeout=timeout_secs*1000)
            infer_time_spent = time() - infer_start_time
            print(f"Inference took {infer_time_spent} seconds")
            # Get output
            output_data = results.as_numpy("OUTPUT_0")
            transcription = output_data[0].decode('utf-8')
            print(f"Transcription for {os.path.basename(wav_file)}: {transcription}")

            # Optionally, save transcription to a file
            output_txt_file = os.path.splitext(wav_file)[0] + ".txt"
            with open(output_txt_file, "w", encoding="utf-8") as f:
                f.write(transcription)
            print(f"Transcription saved to {output_txt_file}")

        except Exception as e:
            infer_time_spent = time() - infer_start_time
            print(f"Error processing {wav_file}: {e}")
            print(f"Inference took {infer_time_spent} seconds")

if __name__ == "__main__":
    main()

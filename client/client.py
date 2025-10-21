import argparse
import glob
import os
from time import time
import librosa
import numpy as np
import tritonclient.http as http_client
import io
import soundfile as sf
from pydub import AudioSegment

MODEL_NAME = "turbo_cuda"
TRITON_SERVER_URL = "localhost:8000"

def preprocess_audio(audio_path, audio_format):
    # Load audio
    try:
        audio, sr = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz
    except ValueError as err:
        print(f"Error loading audio file {audio_path}: {err}")
        # Use pydub to open audio and convert to a standardized WAV in memory
        audio_segment = AudioSegment.from_file(audio_path)
        audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
        
        # Export to a buffer
        buffer = io.BytesIO()
        audio_segment.export(buffer, format="wav")
        buffer.seek(0)
        
        # Load with librosa from the buffer
        audio, sr = librosa.load(buffer, sr=16000)
        print(audio.shape, sr)
        
    length_seconds = len(audio) / sr
    return audio.astype(np.float32), length_seconds

def main():
    parser = argparse.ArgumentParser(description="Triton Whisper Client for WAV files.")
    parser.add_argument("directory", type=str, help="Directory containing .wav files.")
    args = parser.parse_args()

    wav_files = glob.glob(os.path.join(args.directory, "*.wav")) + glob.glob(os.path.join(args.directory, "*.WAV"))
    timeout_secs = 240
    if not wav_files:
        wav_files = []
    mp3_files = glob.glob(os.path.join(args.directory, "*.mp3")) + glob.glob(os.path.join(args.directory, "*.MP3"))
    if not mp3_files:
        mp3_files = []
    wav_files = [(w, 'wav') for w in wav_files if 'part' not in w]
    mp3_files = [(w, 'mp3') for w in mp3_files if 'part' not in w]

    audio_files = wav_files + mp3_files

    print(f"Found {len(audio_files)} files. Sending to Triton server...")

    client = http_client.InferenceServerClient(url=TRITON_SERVER_URL,
        connection_timeout=timeout_secs,
        network_timeout=timeout_secs)
    seconds_transcribed = 0.0
    seconds_of_inference = 0.0
    gpu_hour_cost_dollars = 0.74
    index = 0
    for filepath, file_type in audio_files:
        print(f"Processing {filepath}...")
        try:
            # Preprocess audio
            audio_input_data, length_seconds = preprocess_audio(filepath, file_type)
            
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
            print(f"Length of audio: {length_seconds} seconds")
            if index > 1:
                seconds_transcribed += length_seconds
                seconds_of_inference += infer_time_spent
            speed = length_seconds / infer_time_spent
            print(f"current speed: {speed} seconds per second")

            hour_cost = gpu_hour_cost_dollars / speed
            print(f"hour cost: {hour_cost} dollars per hour of transcription")

            # Get output
            output_data = results.as_numpy("OUTPUT_0")
            transcription = output_data[0].decode('utf-8')
            print(f"Transcription for {os.path.basename(filepath)}: {transcription}")

            # Optionally, save transcription to a file
            output_txt_file = os.path.splitext(filepath)[0] + ".txt"
            with open(output_txt_file, "w", encoding="utf-8") as f:
                f.write(transcription)
            print(f"Transcription saved to {output_txt_file}")

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            
            infer_time_spent = time() - infer_start_time
            print(f"Inference took {infer_time_spent} seconds")
            quit(1)
        index += 1
    print("All files processed.")   
    speed = seconds_transcribed / seconds_of_inference
    print(f"current speed: {speed} seconds per second")

    hour_cost = gpu_hour_cost_dollars / speed
    print(f"hour cost: {hour_cost} dollars per hour of transcription")

if __name__ == "__main__":
    main()

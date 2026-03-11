import time
import torch
from utils import get_inputs, eval_transcriptions
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def load_whisper_cpu(model_name_str, lang="portuguese"):
    from transformers import pipeline
    model = pipeline("automatic-speech-recognition", 
            model=model_name_str,
            return_timestamps=True,
            generate_kwargs={"language": lang},
            device='cpu')
    return model

def process_with_pipeline(model, audio_array):
    result = model(audio_array)
    out_text = result["text"]
    return out_text

def process_low_level(model, processor, audio_array, sr, make_chunks=False):
        
    # Prepare inputs with attention_mask to avoid warnings
    if not make_chunks:
        inputs = processor(
            audio_array, 
            sampling_rate=sr, 
            return_tensors="pt",
            return_attention_mask=True 
        ).to(device)
        
        # 3. Use .generate() natively on the chunk
        with torch.no_grad():
            predicted_ids = model.generate(
                **inputs,
                return_timestamps=True,
                language="portuguese"
            )
        
        # Decode the chunk
        text_chunks = []
        for part in processor.batch_decode(predicted_ids, skip_special_tokens=True):
            text_chunks.append(part.strip())
        chunk_text = " ".join(text_chunks)
    else:
        chunk_length_s = 30
        chunk_samples = chunk_length_s * sr
        
        transcribed_chunks = []
        
        # 2. Manually chunk the audio to process lengths > 30s
        for i in range(0, len(audio_array), chunk_samples):
            chunk = audio_array[i : i + chunk_samples]
            
            # Prepare inputs with attention_mask to avoid warnings
            inputs = processor(
                chunk, 
                sampling_rate=sr, 
                return_tensors="pt",
                return_attention_mask=True 
            ).to(device)
            
            # 3. Use .generate() natively on the chunk
            with torch.no_grad():
                predicted_ids = model.generate(
                    **inputs,
                    return_timestamps=True,
                    language="portuguese"
                )
            
            # Decode the chunk
            chunk_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcribed_chunks.append(chunk_text.strip())
            
        # Join all transcribed chunks for the full audio
        out_text = " ".join(transcribed_chunks)
    return out_text

if __name__ == "__main__":
    # 1. Load the generic processor and model
    model_id = "openai/whisper-base"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_low_level = True
    
    if use_low_level:
        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
    else:
        processor = None
        model = load_whisper_cpu(model_id)

    ds, sample = get_inputs()
    original_scripts = ds["roteiro_segmentado"]

    transcriptions = []
    texts_segmented = []
    time_start = time.time()

    line_n = 0
    for item in tqdm(ds):
        audio = item["audio"]
        audio_array = audio["array"]
        sr = audio["sampling_rate"]
        audio_len_secs = len(audio_array) / sr

        if use_low_level:
            out_text = process_low_level(model, processor, audio_array, 
                sr, make_chunks=True)
        else:
            result = model(audio_array)
            out_text = result["text"]
        
        transcriptions.append(out_text)
        
        original_text = original_scripts[line_n]
        texts_segmented.append(original_text)

        print("Original:")
        print(original_text)
        print("Transcribed:")
        print(out_text)
        print()
        line_n += 1

    time_end = time.time()
    print(f"Total time: {time_end - time_start}")

    texts_joined = []
    for text in texts_segmented:
        lines = []
        for line in text:
            if len(line) == 2:
                speaker, txt = line
                lines.append(txt)
            elif len(line) == 1:
                lines.append(line[0])
            else:
                print("Error: more than 2 elements in line")
                quit(1)

        texts_joined.append(' . '.join(lines))

    for original, transcribed in zip(texts_joined, transcriptions):
        print("Original:")
        print(original)
        print("Transcribed:")
        print(transcribed)
        print()

    final_wer = eval_transcriptions(transcriptions, texts_joined)
    print(f"Final WER: {final_wer}")
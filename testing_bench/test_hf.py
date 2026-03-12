import time
import re
import os
import sys
import torch
import unidecode
from glob import glob
import gc

from tqdm import tqdm
import string

import pandas as pd

from utils import (get_inputs, eval_transcriptions, intelligent_chunk_merge, 
    clean_reference_text, normalize_light, normalize_heavy)


def fix_whisper_generation_config(model, model_name_str):
    """Injects missing generation_config parameters into community models."""
    from transformers import GenerationConfig
    
    # Verifica se a configuração atual está quebrada/incompleta
    if getattr(model.generation_config, "no_timestamps_token_id", None) is None:
        # Tenta adivinhar o modelo base pelo nome
        name_lower = model_name_str.lower()
        base_id = "openai/whisper-base" # fallback padrão
        
        if "tiny" in name_lower: base_id = "openai/whisper-tiny"
        elif "small" in name_lower: base_id = "openai/whisper-small"
        elif "medium" in name_lower: base_id = "openai/whisper-medium"
        elif "large" in name_lower: base_id = "openai/whisper-large-v2"
        
        # Puxa a configuração imaculada da OpenAI e sobrescreve a do modelo problemático
        print(f"Fixing generation_config for {model_name_str} using {base_id} as template...")
        model.generation_config = GenerationConfig.from_pretrained(base_id)
        
    return model

def load_asr_generator(model_name_str, device):
    model_family = "whisper"
    if "mms" in model_name_str:
        model_family = "mms"
    
    if model_family == "whisper":
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        processor = WhisperProcessor.from_pretrained(model_name_str)
        model = WhisperForConditionalGeneration.from_pretrained(model_name_str).to(device)
        
        # Aplica a correção da configuração
        model = fix_whisper_generation_config(model, model_name_str)
        
    elif model_family == "mms":
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name_str)
        model = Wav2Vec2ForCTC.from_pretrained(model_name_str).to(device)
    else:
        from transformers import AutoProcessor, AutoModelForSeq2SeqLM
        processor = AutoProcessor.from_pretrained(model_name_str)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_str).to(device)
        
    return processor, model

def load_whisper(model_name_str, device, lang="portuguese"):
    from transformers import pipeline
    
    if "whisper" in model_name_str.lower():
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        # Para consertar o pipeline, precisamos carregar o modelo explicitamente primeiro
        processor = WhisperProcessor.from_pretrained(model_name_str)
        model_obj = WhisperForConditionalGeneration.from_pretrained(model_name_str)
        
        # Aplica a correção da configuração
        model_obj = fix_whisper_generation_config(model_obj, model_name_str)
        
        # Passa os objetos corrigidos para o pipeline ao invés da string
        model = pipeline(
            "automatic-speech-recognition", 
            model=model_obj,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            return_timestamps=True,
            generate_kwargs={"language": lang},
            device=device
        )
    else:
        model = pipeline(
            "automatic-speech-recognition", 
            model=model_name_str,
            return_timestamps=True,
            generate_kwargs={"language": lang},
            device=device
        )
        
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
        chunk_text = intelligent_chunk_merge(text_chunks)
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
        out_text = intelligent_chunk_merge(transcribed_chunks)
    return out_text

def test_model(model_id, n_samples, device):
    n_for_speedup_test = int(n_samples*0.05)
    if n_for_speedup_test < 5:
        n_for_speedup_test = 5
    model = load_whisper(model_id, device)

    ds, sample = get_inputs(max_n=n_samples)
    original_scripts = ds["roteiro_segmentado"]

    transcriptions = []
    texts_segmented = []
    time_start = time.time()
    audio_lens = []

    line_n = 0
    processing_times = []

    lines_for_low_level_eval = []
    for item in tqdm(ds):
        audio = item["audio"]
        audio_array = audio["array"]
        sr = audio["sampling_rate"]
        audio_len_secs = len(audio_array) / sr

        time_start = time.time()
        
        result = model(audio_array)
        out_text = result["text"]
        processing_times.append(time.time() - time_start)
        
        transcriptions.append(out_text)
        
        original_text = original_scripts[line_n]
        texts_segmented.append(original_text)
        audio_lens.append(audio_len_secs)

        print("Original:")
        print(original_text)
        print("Transcribed:")
        print(out_text)
        print()

        if line_n < n_for_speedup_test:
            lines_for_low_level_eval.append(
                [original_text, audio_array, processing_times[-1]])
        line_n += 1
    
    total_audio_len = sum(audio_lens)
    processing_time = sum(processing_times)
    print(f"Total time: {processing_time}")
    speed = total_audio_len / processing_time
    print(f"Speed: {speed:.2f}x")

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
            
        ref_raw = ' . '.join(lines)
        ref_clean = clean_reference_text(ref_raw)

        texts_joined.append(ref_clean)
    
    texts_original_norm = [normalize_light(t) for t in texts_joined]
    texts_original_simplified = [normalize_heavy(t) for t in texts_joined]

    transcribed_norm = [normalize_light(t) for t in transcriptions]
    transcribed_simplified = [normalize_heavy(t) for t in transcriptions]

    raw_result_lines = []
    iterator = zip(audio_lens, texts_original_norm, texts_original_simplified, 
        transcribed_norm, transcribed_simplified, processing_times)
    for obj in tqdm(iterator):
        audio_len, original_norm, original_simple, transcribed_norm, transcribed_simple, processing_time = obj
        print("Original (norm):")
        print(original_norm)
        print("Original (simple):")
        print(original_simple)
        print("Transcribed (norm):")
        print(transcribed_norm)
        print("Transcribed (simple):")
        print(transcribed_simple)
        print()

        raw_result_lines.append({
            "audio_len": audio_len,
            "processing_time": processing_time,
            "original_norm": original_norm,
            "original_simple": original_simple,
            "transcribed_norm": transcribed_norm,
            "transcribed_simple": transcribed_simple
        })
    
    print("Evaluating speed on splitted audio chunks...")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    processor, model = load_asr_generator(model_id, device)
    new_proc_times = []
    for original_text, audio_array, processing_time in tqdm(lines_for_low_level_eval):
        time_start = time.time()
        out_text = process_low_level(model, processor, audio_array, 
            sr, make_chunks=True)
        processing_time2 = time.time() - time_start
        new_proc_times.append(processing_time2)
    
    full_audio_times = sum([line[-1] for line in lines_for_low_level_eval])
    chunked_audio_times = sum(new_proc_times)
    speedup = full_audio_times / chunked_audio_times
    print(f"Speedup: {speedup:.2f}x")

    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

    return raw_result_lines, speedup
    

if __name__ == "__main__":
    # 1. Load the generic processor and model
    n_samples_to_test = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    
    hf_models = [
        "openai/whisper-base",
        "dominguesm/whisper-tiny-pt",
        "openai/whisper-small",
        "remynd/whisper-small-pt",
        "thiagobarbosa/whisper-base-common-voice-16-pt-v6",
        "openai/whisper-medium",
        "remynd/whisper-medium-pt",
        "pierreguillou/whisper-medium-portuguese",
        "nilc-nlp/distil-whisper-coraa-mupe-asr",
        "openai/whisper-large-v3-turbo",
        "FastFlowLM/Whisper-V3-Turbo-NPU2",
        "inesc-id/WhisperLv3-FT",
        "openai/whisper-large-v3",
        "remynd/whisper-large-v3-pt"
    ]

    eval_csv_paths = glob(os.path.join(results_dir, "eval_results.*.csv"))
    previous_n_samples_of_models = {}
    for prev_csv in eval_csv_paths:
        eval_df = pd.read_csv(prev_csv)
        for _, row in eval_df.iterrows():
            if row['model_id'] not in previous_n_samples_of_models:
                previous_n_samples_of_models[row['model_id']] = row['n_samples']
            else:
                previous_n_samples_of_models[row['model_id']] = max(previous_n_samples_of_models[row['model_id']], row['n_samples'])

    hf_models_to_test = []
    #only test models with fewer samples in the previous evaluation
    for model_id in hf_models:
        if model_id not in previous_n_samples_of_models:
            hf_models_to_test.append(model_id)
        elif previous_n_samples_of_models[model_id] < n_samples_to_test:
            hf_models_to_test.append(model_id)
    hf_models_to_load_from_previous = [m for m in hf_models if m not in hf_models_to_test]
    if len(hf_models_to_test) == 0:
        print("All models have been tested with the specified number of samples.")
        quit(0)
    
    all_raw_results = []

    for model_id in hf_models_to_test:
        raw_result_lines, speedup = test_model(model_id, n_samples_to_test, device)
        for line in raw_result_lines:
            line['model_id'] = model_id
            line['chunking_speedup'] = speedup
        all_raw_results.extend(raw_result_lines)
    
    df = pd.DataFrame(all_raw_results)

    current_result_n = 0
    for p in eval_csv_paths:
        p_n = int(p.split(".")[1])
        if p_n >= current_result_n:
            current_result_n = p_n+1

    raw_results_basename = os.path.join(results_dir, f"raw_results.{current_result_n}")
    eval_basename = os.path.join(results_dir, f"eval_results.{current_result_n}")
    #save csv and excel
    df.to_csv(f"{raw_results_basename}.csv", index=False)
    df.to_excel(f"{raw_results_basename}.xlsx", index=False)
    
    eval_lines = []

    for model_id, model_lines in df.groupby('model_id'):
        transcribed_norm = model_lines['transcribed_norm'].tolist()
        transcribed_simplified = model_lines['transcribed_simple'].tolist()
        texts_original_norm = model_lines['original_norm'].tolist()
        texts_original_simplified = model_lines['original_simple'].tolist()

        wer_norm = eval_transcriptions(transcribed_norm, texts_original_norm)
        wer_simple = eval_transcriptions(transcribed_simplified, texts_original_simplified)
        print(f"Final WER (norm): {wer_norm}")
        print(f"Final WER (simple): {wer_simple}")

        total_audio_seconds = model_lines['audio_len'].sum()
        seconds_sum_original = model_lines['processing_time'].sum()
        seconds_sum_chunked = (model_lines['processing_time'] / model_lines['chunking_speedup']).sum()
        speed_not_optimized = total_audio_seconds / seconds_sum_original
        #In a real scenario, the audio would arrive already chunked.
        #Because of this, we need to estimate what would be the total processing time if the audio was already in small chunks.
        #To do this, we divide the processing time by the speedup factor.
        speed_optimized = total_audio_seconds / seconds_sum_chunked

        eval_lines.append({
            "model_id": model_id,
            "wer_norm": wer_norm,
            "wer_simple": wer_simple,
            "total_audio_seconds": total_audio_seconds,
            "seconds_sum_original": seconds_sum_original,
            "seconds_sum_chunked": seconds_sum_chunked,
            "speed_not_optimized": speed_not_optimized,
            "speed_optimized": speed_optimized,
            "n_samples": len(model_lines)
        })
    
    eval_df = pd.DataFrame(eval_lines)
    eval_df.to_csv(f"{eval_basename}.csv", index=False)
    eval_df.to_excel(f"{eval_basename}.xlsx", index=False)
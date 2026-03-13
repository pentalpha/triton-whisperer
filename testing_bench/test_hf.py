import time
import re
import os
import sys
import torch
import unidecode
from glob import glob
import gc
import traceback

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

'''def load_asr_generator(model_name_str, device):
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
    elif "mms" in model_name_str.lower():
        model = pipeline(
            "automatic-speech-recognition", 
            model=model_name_str,
            return_timestamps='word',
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
        
    return model'''


def process_with_pipeline(model, audio_array):
    result = model(audio_array)
    out_text = result["text"]
    return out_text

def load_asr_generator(model_name_str, device):
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    
    # FIX: Carregar sempre em float16 na GPU para evitar OOM
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model_family = "whisper"
    if "mms" in model_name_str.lower():
        model_family = "mms"
    
    if model_family == "whisper":
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        processor = AutoProcessor.from_pretrained(model_name_str)
        # Injeta o torch_dtype e low_cpu_mem_usage
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name_str, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True
        ).to(device)
        model = fix_whisper_generation_config(model, model_name_str)
        
    elif model_family == "mms":
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name_str)
        model = Wav2Vec2ForCTC.from_pretrained(
            model_name_str, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True
        ).to(device)
    else:
        from transformers import AutoProcessor, AutoModelForSeq2SeqLM
        processor = AutoProcessor.from_pretrained(model_name_str)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_str, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True
        ).to(device)
        
    hf_logging.set_verbosity_warning()
    return processor, model

def load_whisper(model_name_str, device, lang="portuguese"):
    from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
    from transformers.utils import logging as hf_logging
    
    hf_logging.set_verbosity_error()
    name_lower = model_name_str.lower()
    
    # FIX: Precisão para não estourar a VRAM
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    batch_size = 8
    
    if "whisper" in name_lower:
        processor = AutoProcessor.from_pretrained(model_name_str)
        model_obj = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name_str, 
            torch_dtype=torch_dtype, 
            low_cpu_mem_usage=True
        ).to(device)
        model_obj = fix_whisper_generation_config(model_obj, model_name_str)
        
        timestamps_param = "word" if "crisper" in name_lower else True
        
        model = pipeline(
            "automatic-speech-recognition", 
            model=model_obj,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            return_timestamps=timestamps_param,
            chunk_length_s=30,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            # FIX: Adicionado "task": "transcribe" para evitar traduções fantasmas!
            generate_kwargs={"language": lang, "task": "transcribe"},
            device=device,
            language=lang,
            #task="transcribe"
        )
    elif "mms" in name_lower:
        model = pipeline(
            "automatic-speech-recognition", 
            model=model_name_str,
            return_timestamps='word',
            chunk_length_s=30,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            generate_kwargs={"language": lang}, # MMS não usa "task"
            device=device
        )
    else:
        model = pipeline(
            "automatic-speech-recognition", 
            model=model_name_str,
            return_timestamps=True,
            chunk_length_s=30,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            generate_kwargs={"language": lang},
            device=device
        )
        
    hf_logging.set_verbosity_warning()
    return model

def process_low_level(model, processor, audio_array, sr, make_chunks=False):
    from transformers import Wav2Vec2ForCTC
    import torch
    
    is_mms = isinstance(model, Wav2Vec2ForCTC)
    
    if is_mms:
        processor.tokenizer.set_target_lang("por")
        model.load_adapter("por")

    # Identifica o modelo para ajustar as configurações da geração
    try:
        model_name = model.config._name_or_path.lower()
    except:
        model_name = ""
        
    # CrisperWhisper obriga o uso de word-timestamps
    timestamps_param = "word" if "crisper" in model_name else True

    gpu_time_sum = 0.0

    if not make_chunks:
        start_time = time.time()
        inputs = processor(
            audio_array, 
            sampling_rate=sr, 
            return_tensors="pt",
            return_attention_mask=True 
        ).to(model.device) 
        gpu_time_sum += time.time() - start_time
        
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(model.dtype)
        if "input_values" in inputs:
            inputs["input_values"] = inputs["input_values"].to(model.dtype)
        
        with torch.no_grad():
            start_time = time.time()
            if is_mms:
                logits = model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
            else:
                predicted_ids = model.generate(
                    **inputs,
                    return_timestamps=timestamps_param,
                    language="portuguese", task="transcribe"
                )
            gpu_time_sum += time.time() - start_time
        
        text_chunks = []
        for part in processor.batch_decode(predicted_ids, skip_special_tokens=True):
            text_chunks.append(part.strip())
        chunk_text = intelligent_chunk_merge(text_chunks)
        out_text = chunk_text
        
    else:
        chunk_length_s = 30
        chunk_samples = chunk_length_s * sr
        gpu_time_sum = 0.0
        transcribed_chunks = []
        
        for i in range(0, len(audio_array), chunk_samples):
            chunk = audio_array[i : i + chunk_samples]
            
            start_time = time.time()
            inputs = processor(
                chunk, 
                sampling_rate=sr, 
                return_tensors="pt",
                return_attention_mask=True 
            ).to(model.device)
            gpu_time_sum += time.time() - start_time

            if "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(model.dtype)
            if "input_values" in inputs:
                inputs["input_values"] = inputs["input_values"].to(model.dtype)
            
            with torch.no_grad():
                start_time = time.time()
                if is_mms:
                    logits = model(**inputs).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                else:
                    predicted_ids = model.generate(
                        **inputs,
                        return_timestamps=timestamps_param,
                        language="portuguese", task="transcribe"
                    )
                gpu_time_sum += time.time() - start_time
            
            chunk_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcribed_chunks.append(chunk_text.strip())
            
        out_text = intelligent_chunk_merge(transcribed_chunks)
        
    return out_text, gpu_time_sum

def test_model(model_id, n_samples, device):
    print(f"Testing model {model_id} on {n_samples} samples...")
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
        out_text = result["text"].strip()
        if len(out_text) == 0:
            print("Empty transcription")
            raise ValueError("Empty transcription")
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
        out_text, gpu_time = process_low_level(model, processor, audio_array, 
            sr, make_chunks=True)
        processing_time2 = time.time() - time_start
        new_proc_times.append(gpu_time)
    
    full_audio_times = sum([line[-1] for line in lines_for_low_level_eval])
    chunked_audio_times = sum(new_proc_times)
    speedup = full_audio_times / chunked_audio_times
    print(f"Speedup: {speedup:.2f}x")

    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()

    return raw_result_lines, speedup

def save_results(raw_result_lines):
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
    #sort df by wer_norm
    eval_df = eval_df.sort_values(by='wer_norm')
    eval_df.to_csv(f"{eval_basename}.csv", index=False)
    eval_df.to_excel(f"{eval_basename}.xlsx", index=False)

if __name__ == "__main__":
    # 1. Load the generic processor and model
    n_samples_to_test = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    '''"ibm-granite/granite-4.0-1b-speech",
        "Qwen/Qwen3-ASR-0.6B",
        "Qwen/Qwen3-ASR-1.7B",
        "nvidia/canary-1b-flash",
        "nvidia/canary-qwen-2.5b",
        "nvidia/parakeet-tdt-0.6b-v3",
        "nvidia/parakeet-tdt-0.6b-v2",
        "microsoft/Phi-4-multimodal-instruct",'''
    hf_models = [
        #"nyrahealth/CrisperWhisper",
        "facebook/mms-1b-all",
        #"inesc-id/WhisperLv3-FT",
        #"thiagobarbosa/whisper-base-common-voice-16-pt-v6",
        "pierreguillou/whisper-medium-portuguese",
        "nilc-nlp/distil-whisper-coraa-mupe-asr",
        "openai/whisper-large-v3-turbo",
        "openai/whisper-large-v3",
        "remynd/whisper-large-v3-pt",
        "openai/whisper-base",
        "dominguesm/whisper-tiny-pt",
        "openai/whisper-small",
        "remynd/whisper-small-pt",
        "openai/whisper-medium",
        "remynd/whisper-medium-pt",
    ]

    eval_csv_paths = glob(os.path.join(results_dir, "eval_results.*.csv"))
    previous_n_samples_of_models = {}
    for prev_csv in eval_csv_paths:
        eval_df = pd.read_csv(prev_csv)
        for _, row in eval_df.iterrows():
            wer_norm = row['wer_norm']
            if wer_norm > 0.75:
                continue
            else:
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
        try:
            raw_result_lines, speedup = test_model(model_id, n_samples_to_test, device)
            for line in raw_result_lines:
                line['model_id'] = model_id
                line['chunking_speedup'] = speedup
            all_raw_results.extend(raw_result_lines)
            save_results(all_raw_results)
        except Exception as e:
            print(f"Error testing model {model_id}: {e}")
            print(e)
            print(traceback.format_exc())
            continue
    
    
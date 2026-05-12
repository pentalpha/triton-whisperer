import os
import re
import time
import threading
import subprocess
import sys
import codecs

import psutil
import librosa
from flask import Flask, jsonify, request
from qwen_asr import Qwen3ASRModel

app = Flask(__name__)

# ==========================================
# 1. Configurações Globais e Limite de Concorrência
# ==========================================
MAX_CONCURRENT_REQUESTS = 16
semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

# ==========================================
# 2. Carregamento do Modelo (Única Instância)
# ==========================================
print("Iniciando o carregamento do modelo Qwen/Qwen3-ASR-1.7B...")

global_model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    device_map="cuda:0",
    # attn_implementation="flash_attention_2",
    max_inference_batch_size=32,  # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
    max_new_tokens=768,  # Maximum number of tokens to generate. Set a larger value for long audio input.
)
print("Modelo carregado com sucesso na GPU!")


# ==========================================
# 3. Funções Auxiliares de Pós-processamento
# ==========================================
def remove_duplicates_regex(text: str, max_ngram: int = 5) -> str:
    """Remove repetições consecutivas de n-gramas (1..max_ngram)."""
    if not text or text.strip() == "":
        return text

    tokens = re.findall(r"\w+|\W+", text, flags=re.UNICODE)
    word_indices = [
        i for i, tok in enumerate(tokens) if re.match(r"\w+", tok, flags=re.UNICODE)
    ]
    words = [tokens[i] for i in word_indices]
    lower_words = [w.lower() for w in words]

    if not words:
        return text

    keep_word = [True] * len(words)
    i = 0
    L = len(words)

    while i < L:
        matched = False
        max_n = min(max_ngram, L - i)
        for n in range(max_n, 0, -1):
            seq = tuple(lower_words[i : i + n])
            j = i + n
            while j + n <= L and tuple(lower_words[j : j + n]) == seq:
                j += n
            if j > i + n:
                for k in range(i + n, j):
                    keep_word[k] = False
                i = j
                matched = True
                break
        if not matched:
            i += 1

    out = []
    widx = 0
    for idx, tok in enumerate(tokens):
        if re.match(r"\w+", tok, flags=re.UNICODE):
            if keep_word[widx]:
                out.append(tok)
            widx += 1
        else:
            out.append(tok)

    result = "".join(out)
    result = re.sub(r"\s{2,}", " ", result)
    result = re.sub(r"\s+([,.;:!?])", r"\1", result)
    return result.strip()


def remove_duplicates_regex_simple(seq: str) -> str:
    return re.sub(r"\b(\w+)(?:\W+\1\b)+", r"\1", seq, flags=re.IGNORECASE)


# ==========================================
# 4. Rotas da API
# ==========================================
@app.route("/metrics", methods=["GET"])
def gpu():
    try:
        out = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ]
            )
            .decode()
            .strip()
        )
        util, used, total = map(int, out.split(", "))
    except Exception as e:
        util, used, total = 0, 0, 0
        print(f"Erro ao ler nvidia-smi: {e}")

    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem_info = psutil.virtual_memory()

    return jsonify(
        {
            "gpu_utilization_percent": util,
            "vram_used_mb": used,
            "vram_total_mb": total,
            "cpu_usage_percent": cpu_percent,
            "ram_used_bytes": mem_info.used,
            "ram_total_bytes": mem_info.total,
        }
    )


@app.route("/transcribe", methods=["POST"])
def asr_infer():
    # Verifica se um arquivo de áudio foi enviado na requisição POST
    if "audio" not in request.files:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Nenhum arquivo de áudio encontrado. Use o form-data com a chave 'audio'.",
                }
            ),
            400,
        )

    audio_file = request.files["audio"]

    # O Semaphore garante que no máximo 16 threads passem deste ponto simultaneamente.
    # A 17ª ficará em modo de espera (bloqueada) até que uma das 16 termine e chame release().
    with semaphore:
        try:
            # 1. Carregar o áudio e garantir o Sample Rate de 16kHz
            # librosa.load converte automaticamente para mono e faz o resample para sr=16000
            audio_input_data, sr = librosa.load(audio_file, sr=16000)

            # 2. Lógica de chunking do seu código original
            chunk_length_s = 29
            chunk_samples = chunk_length_s * sr

            chunks = [
                (audio_input_data[i : i + chunk_samples], sr)
                for i in range(0, len(audio_input_data), chunk_samples)
            ]

            start_time = time.time()

            result = global_model.transcribe(
                audio=chunks,
                language="Portuguese",
            )
            transcript = " ".join([r.text.strip() for r in result])

            time_spent = time.time() - start_time

            print(type(transcript), repr(transcript), file=sys.stderr)
            print("Raw transcript repr:", repr(transcript), file=sys.stderr)

            # If transcript literally contains escape sequences like \xc9, unescape them
            if isinstance(transcript, str) and r"\x" in transcript:
                print("Scaped str", file=sys.stderr)
                transcript = codecs.decode(transcript, "unicode_escape")
            elif isinstance(transcript, bytes):
                print("bytes str", file=sys.stderr)
                transcript = transcript.decode("utf-8")

            print("Fixed transcript:", transcript, file=sys.stderr)

            print(
                f"Transcrição de {len(chunks)} chunks levou {time_spent:.2f} segundos"
            )

            # 4. Pós-processamento de texto
            transcript = remove_duplicates_regex(transcript)
            transcript = remove_duplicates_regex_simple(transcript)

            return jsonify(
                {
                    "status": "success",
                    "transcription": transcript,
                    "inference_time_seconds": round(time_spent, 2),
                }
            )

        except Exception as e:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": str(e),
                    }
                ),
                500,
            )


if __name__ == "__main__":
    # threaded=True garantindo que requisições simultâneas abram novas threads.
    app.run(host="0.0.0.0", port=8000, threaded=True)

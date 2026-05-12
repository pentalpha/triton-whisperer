import json
from time import sleep
import re
import sys
import os
import base64
import codecs

from time import time
import numpy as np
import triton_python_backend_utils as pb_utils

# --- FIX FOR CIRCULAR IMPORT ---
# Temporarily remove Triton's model directory from sys.path so the 'qwen_asr' 
# package doesn't accidentally import this file when looking for its own 'model.py'.
_current_dir = os.path.dirname(os.path.realpath(__file__))
if _current_dir in sys.path:
    sys.path.remove(_current_dir)

# Now it is safe to import your package
from qwen_asr import Qwen3ASRModel

# Restore the directory back to sys.path
sys.path.insert(0, _current_dir)
# -------------------------------


class TritonPythonModel:
    def initialize(self, args):
        self.qwen_model_config = model_config = json.loads(args['model_config'])
        output_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT_0"
        )
        self.output_dtype = pb_utils.triton_to_numpy_type(
            output_config['data_type']
        )

        self.whisper_mname = "Qwen/Qwen3-ASR-1.7B"
        self.language = "pt"

        print(f"Loading qwen model {self.whisper_mname} with language {self.language}...")
        
        print('whisper_worker_process: Starting model loading')
        
        self.qwen_model = Qwen3ASRModel.from_pretrained(
            self.whisper_mname,
            device_map="cuda:0",
            # attn_implementation="flash_attention_2",
            max_inference_batch_size=32, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
            max_new_tokens=768, # Maximum number of tokens to generate. Set a larger value for long audio input.
        )
        print("Loaded qwen model!")

    def execute(self, requests):
        responses = []
        for request in requests:
            input_audio_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_0")
            audio_input_data = input_audio_tensor.as_numpy()

            chunk_length_s = 30
            sr = 16000
            chunk_samples = chunk_length_s * sr

            chunks = [(audio_input_data[i : i + chunk_samples], sr) 
                      for i in range(0, len(audio_input_data), chunk_samples)]
            start_time = time()
            result = self.qwen_model.transcribe(
                audio=chunks,
                language="Portuguese",
            )
            
            time_spent = time() - start_time
            print(f"Transcription took {time_spent} seconds")
            transcript = ' '.join([r.text.strip() for r in result])
            transcript = self._remove_duplicates_regex(transcript)
            transcript = self._remove_duplicates_regex_simple(transcript)
            
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
            encoded = transcript.encode("utf-8")

            output_transcript = np.frombuffer(
                encoded,
                dtype=np.uint8
            ).copy()

            print(output_transcript, file=sys.stderr)
            print(output_transcript.shape, file=sys.stderr)
            print(output_transcript.dtype, file=sys.stderr)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("OUTPUT_0", output_transcript)
                ]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        print('Cleaning up...')

    def _remove_duplicates_regex(self, text: str, max_ngram: int = 5) -> str:
        """
        Remove repetições consecutivas de n-gramas (1..max_ngram).
        Ex.: "fez a de fez a de fez a de" -> "fez a de"
        Mantém separadores (pontuação/espacos) na medida do possível.
        """
        if not text or text.strip() == "":
            return text

        # Tokeniza em palavras (\w+) e não-palavras (separadores)
        tokens = re.findall(r'\w+|\W+', text, flags=re.UNICODE)
        word_indices = [i for i, tok in enumerate(tokens) if re.match(r'\w+', tok, flags=re.UNICODE)]
        words = [tokens[i] for i in word_indices]
        lower_words = [w.lower() for w in words]

        if not words:
            return text

        keep_word = [True] * len(words)
        i = 0
        L = len(words)

        while i < L:
            matched = False
            # tenta maiores n-grams primeiro
            max_n = min(max_ngram, L - i)
            for n in range(max_n, 0, -1):
                seq = tuple(lower_words[i:i + n])
                j = i + n
                # conta quantas vezes a seq se repete consecutivamente
                while j + n <= L and tuple(lower_words[j:j + n]) == seq:
                    j += n
                if j > i + n:
                    # houve repetição: marca palavras repetidas para remoção
                    for k in range(i + n, j):
                        keep_word[k] = False
                    i = j  # pula bloco repetido
                    matched = True
                    break
            if not matched:
                i += 1

        # Reconstrói texto: mantém separadores e apenas palavras marcadas
        out = []
        widx = 0
        for idx, tok in enumerate(tokens):
            if re.match(r'\w+', tok, flags=re.UNICODE):
                if keep_word[widx]:
                    out.append(tok)
                # se palavra removida, não append; mantemos separadores seguintes normalmente
                widx += 1
            else:
                out.append(tok)

        result = ''.join(out)
        # Normaliza espaços extras introduzidos pela remoção
        result = re.sub(r'\s{2,}', ' ', result)
        # Remove espaço antes de pontuação (opcional, melhora saída)
        result = re.sub(r'\s+([,.;:!?])', r'\1', result)
        return result.strip()

    def _remove_duplicates_regex_simple(self, seq):
        my_output = re.sub(r'\b(\w+)(?:\W+\1\b)+', r'\1', seq, flags=re.IGNORECASE)
        return my_output
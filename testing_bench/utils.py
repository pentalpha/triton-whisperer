
import re
import unidecode
import string
import time

from datasets import load_dataset, Audio

colunas = [
    "ID",
    "audio",
    "modelo_audio",
    "roteiro_segmentado",
]

hf_dataset = "pitagoras-alves/fake-emergencies-br"

def get_inputs(max_n=4):
    print("Downloading dataset...")
    ds = load_dataset(
        hf_dataset,
        split=f"train[:{max_n}]",
        streaming=False,
        columns=colunas,
    )#.take(max_n)
    print("Casting audio column...")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    sample = next(iter(ds))["audio"]["array"]
    print("Sample:", sample.shape)

    return ds, sample

def eval_transcriptions(pred, true):
    from evaluate import load
    wer_metric = load("wer")
    return wer_metric.compute(references=true, predictions=pred)

def normalize_light(text):
    text = text.lower()
    # Limpa espaços extras
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_heavy(text):
    text = text.lower()
    # Remover caracteres especiais
    text = unidecode.unidecode(text)
    
    # FIX: Em vez de apenas remover a pontuação, substitua por espaço!
    # Isso impede que falhas de espaçamento como "mordido?nao" virem "mordidonao"
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    
    # Limpa espaços extras
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def intelligent_chunk_merge(chunks):
    if not chunks:
        return ""
    
    merged_text = chunks[0].strip()
    
    for i in range(1, len(chunks)):
        prev = merged_text.strip()
        curr = chunks[i].strip()
        
        # 1. Remove pontuações geradas por cortes abruptos nas bordas
        prev_clean = re.sub(r'[\.\,\;\:\-\…]+$', '', prev)
        curr_clean = re.sub(r'^[\.\,\;\:\-\…]+', '', curr)
        
        prev_words = prev_clean.split()
        curr_words = curr_clean.split()
        
        if prev_words and curr_words:
            last_word = prev_words[-1].lower()
            first_word = curr_words[0].lower()
            
            # 2. Trata duplicações exatas
            if last_word == first_word:
                curr_words.pop(0)
                
            # 3. Trata palavras fragmentadas pelo corte
            elif first_word.startswith(last_word) or last_word.startswith(first_word):
                if len(first_word) > len(last_word):
                    prev_words.pop()
                else:
                    curr_words.pop(0)
                    
        # Junta o texto limpo
        prev_str = " ".join(prev_words)
        curr_str = " ".join(curr_words)
        
        # 4. Ajusta a capitalização para decidir se foi uma quebra de frase
        if curr_str:
            if curr_str[0].islower():
                merged_text = prev_str + " " + curr_str
            else:
                if prev and prev[-1] in ".!?":
                    merged_text = prev_str + prev[-1] + " " + curr_str
                else:
                    merged_text = prev_str + ". " + curr_str
        else:
            merged_text = prev_str

    # 5. Varredura final de limpeza
    # Remove espaço antes de pontuação
    merged_text = re.sub(r'\s+([.,?!])', r'\1', merged_text) 
    
    # FIX: Garante que toda pontuação seja seguida de espaço se a próxima letra for texto
    # Resolve os casos como: "acontecendo?Ajudar!" -> "acontecendo? Ajudar!"
    merged_text = re.sub(r'([.!?,\:])([a-zA-ZÀ-ÿ])', r'\1 \2', merged_text)
    
    # FIX: Separa palavras coladas alucinadas pelo modelo em CamelCase 
    # Resolve os casos como: "seVocê" -> "se Você"
    merged_text = re.sub(r'([a-zà-ÿ])([A-ZÀ-Ÿ])', r'\1 \2', merged_text)
    
    # Limpa espaços extras que possam ter sido gerados nas etapas acima
    merged_text = re.sub(r'\s+', ' ', merged_text)
    
    return merged_text.strip()

def clean_reference_text(text):
    # 1. Remove o artefato de junção " . " (espaço, ponto, espaço) ou pontos isolados no final
    clean_text = re.sub(r'\s+\.\s+', ' ', text)
    clean_text = re.sub(r'\s+\.$', '', clean_text)
    
    # 2. Corrige pontuações duplas (ex: "! ." vira apenas "!")
    clean_text = re.sub(r'([?!,])\s*\.', r'\1', clean_text)
    
    # 3. Remove as reticências (...) que marcam hesitação
    # Substituímos por um espaço para evitar colar palavras (ex: "A...a gente" -> "A a gente")
    clean_text = re.sub(r'\.{2,}', ' ', clean_text)
    
    # 4. Remove múltiplos espaços gerados pelas substituições anteriores
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    return clean_text.strip()
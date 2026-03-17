
import re
import unidecode
import string
import time
import os

from datasets import load_dataset, Audio, load_from_disk

colunas = [
    "ID",
    "audio",
    "modelo_audio",
    "roteiro_segmentado",
]

hf_dataset = "pitagoras-alves/fake-emergencies-br"

def get_inputs(max_n=4):
    #Only load if a file with the correct sampling has not been sampled and saved yet
    df_path = f"inputs/df_{max_n}.parquet"
    if os.path.exists(df_path):
        print(f"Loading from {df_path}")
        ds = load_from_disk(df_path)
        return ds
    else:
        print("Downloading dataset...")
        ds = load_dataset(
            hf_dataset,
            split="train",
            streaming=False,
            columns=colunas,
        )#.take(max_n)
        ds = ds.shuffle(seed=1337)
        ds = ds.select(range(max_n))
        print("Casting audio column...")
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        sample = next(iter(ds))["audio"]["array"]
        print("Sample:", sample.shape)
        os.makedirs("inputs", exist_ok=True)
        ds.save_to_disk(df_path)
        return ds

def eval_transcriptions(pred, true):
    from evaluate import load
    wer_metric = load("wer")
    return wer_metric.compute(references=true, predictions=pred)

def normalize_light(text):
    text = text.lower()
    # Limpa espaços extras
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_special(text):
    text = text.lower()
    # Remover caracteres especiais
    text = unidecode.unidecode(text)
    # Limpa espaços extras
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_punct(text):
    text = text.lower()
    
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

def remove_duplicates_regex(text: str, max_ngram: int = 5) -> str:
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

def normalize_repeats(seq):
    no_dups1 = remove_duplicates_regex(seq)
    no_dups2 = re.sub(r'\b(\w+)(?:\W+\1\b)+', r'\1', no_dups1, flags=re.IGNORECASE)
    return no_dups2
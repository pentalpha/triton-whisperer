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
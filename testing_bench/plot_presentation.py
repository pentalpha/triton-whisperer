import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_presentation.py <results_path>")
        sys.exit(1)
    results_path = sys.argv[1]
    df = pd.read_csv(results_path)
    #wer_normalized -> Word Error Rate
    df["Word Error Rate (WER)"] = df["wer_normalized"] * 100
    df["Nome do Modelo"] = df["model_id"]

    #turn wer_no_special, wer_no_punct, wer_no_repeats into percentage of wer_normalized
    df["Porcentagem do WER: Caracteres especiais incorretos (%)"] = (df["wer_normalized"] - df["wer_no_special"]) / df["wer_normalized"] * 100
    df["Porcentagem do WER: Pontuação incorreta (%)"] = (df["wer_normalized"] - df["wer_no_punct"]) / df["wer_normalized"] * 100
    df["Porcentagem do WER: Repetições (%)"] = (df["wer_normalized"] - df["wer_no_repeats"]) / df["wer_normalized"] * 100

    df["Horas de áudio testadas"] = df["total_audio_seconds"] / 3600
    df["Horas de processamento"] = df["total_seconds_chunks"] / 3600
    df["Velocidade"] = df["speed_chunks"]
    df["Custo 1h Áudio ($)"] = df["cost_1h_demand"]
    df["Custo Total Teste ($)"] = df["cost_test_demand"]

    final_cols = ["Nome do Modelo", "Word Error Rate (WER)", 
        "Porcentagem do WER: Caracteres especiais incorretos (%)", 
        "Porcentagem do WER: Pontuação incorreta (%)", 
        "Porcentagem do WER: Repetições (%)", 
        "Horas de áudio testadas", "Horas de processamento", "Velocidade", 
        "Custo 1h Áudio ($)", "Custo Total Teste ($)"]
    df = df[final_cols]
    original_dir = os.path.dirname(results_path)
    df.to_csv(f"{original_dir}/eval_results_final.csv", index=False)

    #save excel file
    df.to_excel(f"{original_dir}/eval_results_final.xlsx", index=False)
    
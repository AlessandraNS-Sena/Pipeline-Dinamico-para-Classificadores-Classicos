import questionary
import pandas as pd
import sys
from src.engine import PipelineEngine

def main():
    print("\n" + "="*40)
    print("      PIPELINE DE SENTIMENTOS INTERATIVO")
    print("="*40 + "\n")

    # 1. Carregamento do Dataset
    path = questionary.text("Caminho para o arquivo CSV:").ask()
    try:
        df = pd.read_csv(path)
        print(f"Dataset carregado com sucesso! Colunas encontradas: {list(df.columns)}")
    except Exception as e:
        print(f"Erro ao carregar arquivo: {e}")
        sys.exit(1)

    # 2. Configuração das Colunas
    text_col = questionary.text("Nome da coluna de TEXTO:", default="text").ask()
    target_col = questionary.text("Nome da coluna de TARGET (rótulo):", default="label").ask()

    # 3. Lógica de Binarização (Caso seja o B2W ou similar)
    unique_vals = df[target_col].nunique()
    if unique_vals > 2:
        if questionary.confirm("Detectadas mais de 2 classes. Deseja binarizar o target?").ask():
            # Exemplo: Notas 1-2 viram 0 (negativo), 4-5 viram 1 (positivo). 3 é descartado.
            df = df[df[target_col] != 3]  # Remove neutros
            df[target_col] = df[target_col].apply(lambda x: 1 if x >= 4 else 0)
            print("Target binarizado: [0] Negativo (notas 1-2) | [1] Positivo (notas 4-5)")

    # 4. Escolhas de Pré-processamento
    prep_norm = questionary.select(
        "Escolha a técnica de normalização:",
        choices=["stemming", "None"]
    ).ask()

    prep_stop = questionary.select(
        "Estratégia de Stopwords:",
        choices=[
            {"name": "Remover todas", "value": True},
            {"name": "Manter negações (Ex: 'não', 'nunca')", "value": "keep_negations"},
            {"name": "Não remover", "value": False}
        ]
    ).ask()

    handle_neg = questionary.confirm("Deseja aplicar sufixo _NEG em palavras após negação?").ask()
    norm_emoji = questionary.confirm("Deseja converter Emojis para texto?").ask()

    # 5. Escolhas de Vetorização e Modelo
    vec_choice = questionary.select(
        "Escolha a estratégia de vetorização:",
        choices=["BoW", "TF-IDF", "TF-IDF+SVD"]
    ).ask()

    model_choice = questionary.select(
        "Qual classificador deseja treinar?",
        choices=[
            {"name": "Regressão Logística", "value": "logistic_regression"},
            {"name": "Random Forest", "value": "random_forest"}
        ]
    ).ask()

    # 6. Execução do Pipeline
    print("\nIniciando Pipeline... Aguarde os resultados.\n")
    
    engine = PipelineEngine(df, text_col, target_col)
    
    prep_cfg = {
        "lowercase": True,
        "remove_urls": True,
        "remove_stopwords": prep_stop,
        "normalization": prep_norm if prep_norm != "None" else None,
        "handle_negations": handle_neg,
        "normalize_emojis": norm_emoji
    }
    
    vec_cfg = {
        "strategy": vec_choice, 
        "params": {"n_components": 100 if vec_choice == "TF-IDF+SVD" else None}
    }
    
    clf_cfg = {
        "model": model_choice, 
        "mode": "optuna" # Busca automática de hiperparâmetros
    }

    engine.run(prep_cfg, vec_cfg, clf_cfg)

if __name__ == "__main__":
    main()
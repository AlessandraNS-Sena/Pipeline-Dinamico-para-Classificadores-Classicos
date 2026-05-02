from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from src.engine import PipelineEngine

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_pipeline', methods=['POST'])
def run_pipeline():
    try:
        # 1. Recebimento dos arquivos e dados básicos
        file = request.files['dataset']
        model_type = request.form.get('model')
        strategy = request.form.get('strategy')

        if not file:
            return jsonify({"status": "error", "message": "Nenhum arquivo enviado."}), 400

        # 2. Salvamento do arquivo
        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)

        # 3. Leitura Robusta (Resolve erro de tokenização/separador)
        # O uso de sep=None e engine='python' detecta automaticamente o delimitador
        df = pd.read_csv(path, sep=None, engine='python', on_bad_lines='skip')

        # 4. Detecção Inteligente de Colunas (Invisível ao usuário)
        cols = df.columns.tolist()
        if len(cols) < 2:
            return jsonify({"status": "error", "message": "Dataset precisa ter pelo menos 2 colunas."}), 400

        # Heurística inicial por posição
        text_col = cols[0]
        target_col = cols[1]

        # Refinamento por nome (Heurística de Negócio)
        for c in cols:
            cl = str(c).lower()
            if cl in ['text', 'review', 'comentario', 'texto', 'txt']:
                text_col = c
            if cl in ['label', 'target', 'sentiment', 'sentimento', 'rating', 'class']:
                target_col = c

        # 5. Limpeza de Dados (Resolve erro de NaN e [None, None])
        # Primeiro garantimos que as colunas existem, depois limpamos
        df = df.dropna(subset=[text_col, target_col])
        
        # Converte para string para o Preprocessor não falhar com números
        df[text_col] = df[text_col].astype(str)
        
        # Filtra apenas classes que possuem pelo menos 3 exemplos
        df = df[df.groupby(target_col)[target_col].transform('count') >= 3]

        if df.empty:
            return jsonify({"status": "error", "message": "Dataset não possui exemplos suficientes por classe."}), 400
        

        # 6. Configurações do Pipeline (Conforme o Objetivo do Projeto)
        prep_cfg = {
            "normalization": "stemming",
            "handle_negations": True,
            "lowercase": True,
            "remove_stopwords": "keep_negations", # Preserva 'não', 'nunca', etc.
            "normalize_emojis": True
        }

        vec_cfg = {
            "strategy": strategy,
            "params": {
                "n_components": 100 if strategy == "TF-IDF+SVD" else None
            }
        }

        clf_cfg = {
            "model": model_type,
            "mode": "optuna" # Busca automatizada via Optuna
        }

        # 7. Execução do Orquestrador (Engine)
        engine = PipelineEngine(df, text_col, target_col)
        results = engine.run(prep_cfg, vec_cfg, clf_cfg)

        return jsonify({
            "status": "success",
            "results": results,
            "detected_cols": {
                "text_col": text_col,
                "target_col": target_col
            }
        })

    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": f"Erro na execução: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
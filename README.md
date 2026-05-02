# Pipeline Dinâmico de Análise de Sentimentos

&emsp; Este projeto implementa um pipeline dinâmico, modular e configurável para análise de sentimentos em textos, desenvolvido como parte do Módulo 06 do curso de Sistemas de Informação no  Inteli. O sistema automatiza o ciclo completo de ciência de dados: do carregamento de dados brutos à seleção do melhor modelo via Otimização Bayesiana.

## Como Executar a Solução

### 1. Instalação

&emsp; Certifique-se de ter o **Python 3.13+** instalado. Clone o repositório e instale as dependências:

```
git clone https://github.com/seu-usuario/meu-metodo.git
pip install -r requirements.txt
```

### 2. Inicialização de Recursos (NLP)

&emsp; Execute o comando abaixo para baixar os pacotes necessários do NLTK:

```
python -c "import nltk; nltk.download(['punkt', 'stopwords', 'rslp'])"
```

### 3. Execução

&emsp; Inicie o servidor Flask:

```
python app.py
```

&emsp; Acesse no seu navegador: `[http://127.0.0.1:5000](http://127.0.0.1:5000)`

---

## Diferenciais de Engenharia de Dados

### 1. Robustez e Inteligência "Zero-Config"

&emsp; Este pipeline foi projetado para ser resiliente a falhas comuns em datasets reais de larga escala graças aos seguintes pontos:

- Detecção Automática de Colunas: Identifica dinamicamente as colunas de texto e target (rótulo), reduzindo a fricção do usuário.
- Data Cleaning Automático: Tratamento de valores nulos (`NaN`), conversão forçada de tipos para string e descarte de linhas corrompidas (`on_bad_lines='skip'`).
- Auto-Separador : Detecta automaticamente se o CSV utiliza vírgula ou ponto e vírgula.

### 2. Otimização Bayesiana com Optuna

&emsp; O projeto utiliza a biblioteca Optuna para realizar uma busca inteligente no espaço de hiperparâmetros, focando na métrica  F1-macro. Esta abordagem é superior ao GridSearch por ser mais rápida e encontrar melhores resultados em menos iterações.

---

## Análise Crítica dos Experimentos

&emsp; Durante o desenvolvimento e testes com datasets como B2W e  HatEval , foram observados pontos cruciais para a análise crítica do pipeline:

### A. Compatibilidade Teórica: Naive Bayes vs. Redução de Dimensionalidade

&emsp; Um aprendizado técnico importante foi o conflito entre o MultinomialNB e o  TruncatedSVD.

- Problema: O SVD gera componentes que podem ser negativos. O Naive Bayes Multinomial, matematicamente fundamentado em contagens, não aceita valores negativos.
- Solução: O pipeline foi ajustado para que modelos lineares (como LinearSVC) ou baseados em árvores (LightGBM) sejam priorizados quando há redução de dimensionalidade ou uso de embeddings.

### B. O Impacto do `Handle Negations`

&emsp; A técnica de sufixar tokens após negações (ex: "não gosto" para "gosto_NEG") mostrou-se essencial para modelos lineares. Sem isso, o modelo trata "gosto" e "não gosto" com pesos similares, perdendo a semântica de inversão de sentimento.

### C. Estratificação e Classes Raras

&emsp; O erro de classes com apenas 1 membro revelou a necessidade de uma abordagem diferente na divisão de dados. O pipeline prioriza a Estratificação (70/15/15) para manter a distribuição de classes fiel, mas implementa um tratamento de exceção para garantir a execução mesmo em datasets desbalanceados.

### D. Seleção de Métrica: Por que F1-Macro?

&emsp; Optei pelo F1-macro como métrica de otimização pois, em análise de sentimentos (especialmente discurso de ódio), a acurácia pode ser enganosa se uma classe for predominante. O F1-macro garante que o modelo seja penalizado se ignorar a classe minoritária.

---

## Estrutura do Repositório

* app.py: Interface web e rotas da API com lógica de limpeza de dados.
* src/engine.py: Orquestrador da divisão de dados e fluxo do pipeline.
* src/preprocessor.py: Lógica de normalização, negações e emojis.
* src/vectorizer.py: Estratégias de BoW, TF-IDF, SVD e Word2Vec.
* src/classifier.py: Gerenciamento de modelos e busca de hiperparâmetros.

---

Autora: Alessandra Nascimento Santos Sena

Especialização: Sistemas de Informação (Inteli)

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import emoji

class Preprocessor:
    """
    Classe responsável pela limpeza e normalização do texto.
    Implementa técnicas avançadas como tratamento de negação e preservação de stopwords contextuais.
    """
    def __init__(self, config: dict):
        self.config = config
        self.stemmer = RSLPStemmer() # Stemmer específico para a língua portuguesa
        
        # Inicializa a lista padrão de stopwords do NLTK
        self.stop_words = set(stopwords.words('portuguese'))
        
        # Lógica "keep_negations": Essencial para análise de sentimentos.
        # Remove palavras negativas da lista de exclusão para que o modelo perceba inversões de sentido.
        if config.get("remove_stopwords") == "keep_negations":
            negations = {"não", "nunca", "jamais", "nem", "tampouco", "nada"}
            self.stop_words = self.stop_words - negations

    def _handle_negations(self, tokens):
        """
        Técnica de marcação de escopo: Adiciona o sufixo _NEG às palavras que seguem uma negação
        até encontrar uma pontuação, ajudando o vetorizador a distinguir contextos opostos.
        """
        new_tokens = []
        negate = False
        negation_tokens = {"não", "nunca", "jamais", "nem", "tampouco"}
        break_marks = {".", ",", "!", "?", ";"} # Marcas que encerram o efeito da negação
        
        for t in tokens:
            if t in negation_tokens:
                negate = True
                new_tokens.append(t)
                continue
            if t in break_marks:
                negate = False
                new_tokens.append(t)
                continue
            # Transforma "bom" em "bom_NEG" caso esteja sob efeito de uma negação
            new_tokens.append(f"{t}_NEG" if negate else t)
        return new_tokens

    def transform(self, text):
        """
        Executa o pipeline de transformação baseado no dicionário de configuração recebido.
        """
        # Proteção contra dados não textuais (importante para robustez em datasets reais)
        if not isinstance(text, str): return ""
        
        # Limpezas básicas e Regex
        if self.config.get("lowercase"): text = text.lower()
        if self.config.get("remove_urls"): text = re.sub(r'http\S+', '', text)
        
        # Converte emojis em texto (ex: 😄 -> :sorriso:) para que sejam vetorizados
        if self.config.get("normalize_emojis"): text = emoji.demojize(text, language='pt')
        
        # Tokenização: Divide a string em uma lista de palavras/símbolos
        tokens = nltk.word_tokenize(text)
        
        # Filtros de conteúdo
        if self.config.get("remove_stopwords"):
            tokens = [t for t in tokens if t not in self.stop_words]
        
        if self.config.get("handle_negations"):
            tokens = self._handle_negations(tokens)
            
        # Normalização morfológica: Reduz as palavras aos seus radicais
        if self.config.get("normalization") == "stemming":
            tokens = [self.stemmer.stem(t) for t in tokens]
            
        # Reconstrói a string para ser enviada ao Vetorizador
        return " ".join(tokens)
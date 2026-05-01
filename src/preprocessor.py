import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
import emoji

class Preprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.stemmer = RSLPStemmer()
        
        # Lógica de Stopwords
        self.stop_words = set(stopwords.words('portuguese'))
        if config.get("remove_stopwords") == "keep_negations":
            negations = {"não", "nunca", "jamais", "nem", "tampouco", "nada"}
            self.stop_words = self.stop_words - negations

    def _handle_negations(self, tokens):
        new_tokens = []
        negate = False
        negation_tokens = {"não", "nunca", "jamais", "nem", "tampouco"}
        break_marks = {".", ",", "!", "?", ";"}
        
        for t in tokens:
            if t in negation_tokens:
                negate = True
                new_tokens.append(t)
                continue
            if t in break_marks:
                negate = False
                new_tokens.append(t)
                continue
            new_tokens.append(f"{t}_NEG" if negate else t)
        return new_tokens

    def transform(self, text):
        if not isinstance(text, str): return ""
        if self.config.get("lowercase"): text = text.lower()
        if self.config.get("remove_urls"): text = re.sub(r'http\S+', '', text)
        if self.config.get("normalize_emojis"): text = emoji.demojize(text, language='pt')
        
        tokens = nltk.word_tokenize(text)
        if self.config.get("remove_stopwords"):
            tokens = [t for t in tokens if t not in self.stop_words]
        if self.config.get("handle_negations"):
            tokens = self._handle_negations(tokens)
        if self.config.get("normalization") == "stemming":
            tokens = [self.stemmer.stem(t) for t in tokens]
            
        return " ".join(tokens)
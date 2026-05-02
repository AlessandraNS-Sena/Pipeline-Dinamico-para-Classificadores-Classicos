import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from gensim.models import Word2Vec

class Vectorizer:
    def __init__(self, strategy: str, params: dict):
        self.strategy = strategy
        self.params = params
        self.vectorizer = self._build()

    def _build(self):
        max_f = self.params.get("max_features", 5000)
        
        if self.strategy == "BoW":
            return CountVectorizer(
                binary=self.params.get("binary", False), 
                max_features=max_f
            )
        
        elif self.strategy == "TF-IDF":
            return TfidfVectorizer(
                sublinear_tf=self.params.get("sublinear_tf", True),
                ngram_range=self.params.get("ngram_range", (1, 1)),
                max_features=max_f,
                norm=self.params.get("norm", "l2")
            )
        
        elif self.strategy == "TF-IDF+SVD":
            n_comp = self.params.get("n_components", 100)
            return Pipeline([
                ('tfidf', TfidfVectorizer(max_features=max_f)),
                ('svd', TruncatedSVD(n_components=n_comp))
            ])
        
        elif self.strategy == "Word2Vec":
            # Retorna None pois o Word2Vec precisa ser treinado de forma diferente
            return None 

    # Método auxiliar para transformar texto em média de vetores (Word2Vec)
    def word2vec_transform(self, tokens_list, model):
        vectors = [
            np.mean([model.wv[word] for word in tokens if word in model.wv] 
                    or [np.zeros(model.vector_size)], axis=0)
            for tokens in tokens_list
        ]
        return np.array(vectors)
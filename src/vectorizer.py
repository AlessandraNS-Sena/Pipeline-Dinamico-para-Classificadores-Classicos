from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

class Vectorizer:
    def __init__(self, strategy: str, params: dict):
        self.strategy = strategy
        self.params = params
        self.vectorizer = self._build()

    def _build(self):
        max_f = self.params.get("max_features", 5000)
        if self.strategy == "BoW":
            return CountVectorizer(binary=self.params.get("binary", False), max_features=max_f)
        elif self.strategy == "TF-IDF":
            return TfidfVectorizer(sublinear_tf=True, max_features=max_f)
        elif self.strategy == "TF-IDF+SVD":
            return Pipeline([
                ('tfidf', TfidfVectorizer(max_features=max_f)),
                ('svd', TruncatedSVD(n_components=self.params.get("n_components", 100)))
            ])
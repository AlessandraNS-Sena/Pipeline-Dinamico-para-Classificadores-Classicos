from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from .preprocessor import Preprocessor
from .vectorizer import Vectorizer
from .classifier import Classifier

class PipelineEngine:
    def __init__(self, df, text_col, target_col):
        self.df = df
        self.text_col = text_col
        self.target_col = target_col

    def run(self, prep_cfg, vec_cfg, clf_cfg):
        # Split 70/15/15
        train, temp = train_test_split(self.df, test_size=0.3, stratify=self.df[self.target_col])
        val, test = train_test_split(temp, test_size=0.5, stratify=temp[self.target_col])

        # Prep & Vectorize
        prep = Preprocessor(prep_cfg)
        vec = Vectorizer(vec_cfg['strategy'], vec_cfg['params'])
        
        X_train = vec.vectorizer.fit_transform(train[self.text_col].apply(prep.transform))
        X_val = vec.vectorizer.transform(val[self.text_col].apply(prep.transform))
        
        # Train
        clf = Classifier(clf_cfg['model'], clf_cfg['mode'])
        clf.train(X_train, train[self.target_col], X_val, val[self.target_col])
        
        # Test Final
        X_test = vec.vectorizer.transform(test[self.text_col].apply(prep.transform))
        print(classification_report(test[self.target_col], clf.best_model.predict(X_test)))
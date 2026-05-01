from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import optuna

class Classifier:
    def __init__(self, model_type, search_strategy="manual", params=None):
        self.model_type = model_type
        self.search_strategy = search_strategy
        self.params = params or {}
        self.best_model = None

    def _get_model(self, p):
        if self.model_type == "logistic_regression": return LogisticRegression(**p, max_iter=1000)
        if self.model_type == "random_forest": return RandomForestClassifier(**p)

    def train(self, X_train, y_train, X_val, y_val):
        if self.search_strategy == "optuna":
            def objective(trial):
                if self.model_type == "logistic_regression":
                    p = {'C': trial.suggest_float('C', 0.1, 10.0)}
                else:
                    p = {'n_estimators': trial.suggest_int('n_estimators', 50, 200)}
                
                model = self._get_model(p)
                model.fit(X_train, y_train)
                return f1_score(y_val, model.predict(X_val), average='macro')

            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=10)
            self.params = study.best_params
            
        self.best_model = self._get_model(self.params)
        self.best_model.fit(X_train, y_train)
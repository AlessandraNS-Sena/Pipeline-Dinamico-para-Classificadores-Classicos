from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import f1_score
import optuna

class Classifier:
    """
    Classe que encapsula múltiplos classificadores e gerencia a otimização 
    de hiperparâmetros de forma automatizada (Optuna) ou manual.
    """
    def __init__(self, model_type, mode="manual", params=None):
        self.model_type = model_type
        self.mode = mode
        self.params = params or {}
        self.best_model = None

    def _get_model(self, p, vectorization_type="tfidf"):
        """
        Fábrica de modelos que instancia o classificador com base no tipo e hiperparâmetros.
        """
        if self.model_type == "naive_bayes":
            # Escolha estratégica: Bernoulli para features binárias e Multinomial para frequências
            return BernoulliNB(**p) if vectorization_type == "bow_binary" else MultinomialNB(**p)
        
        elif self.model_type == "logistic_regression":
            # max_iter aumentado para garantir convergência em datasets maiores
            return LogisticRegression(**p, max_iter=1000)
        
        elif self.model_type == "linear_svc":
            return LinearSVC(**p, max_iter=2000)
        
        elif self.model_type == "random_forest":
            return RandomForestClassifier(**p)
        
        elif self.model_type == "lightgbm":
            # verbosity=-1 para manter os logs limpos durante a otimização
            return lgb.LGBMClassifier(**p, verbosity=-1)

    def train(self, X_train, y_train, X_val, y_val, vectorization_type="tfidf"):
        """
        Gerencia o fluxo de treinamento. Se o modo for 'optuna', realiza a busca 
        pelos melhores hiperparâmetros antes do ajuste final.
        """
        if self.mode == "optuna":
            def objective(trial):
                # Espaço de busca definido para cada algoritmo conforme boas práticas de mercado
                p = {}
                if self.model_type == "logistic_regression":
                    p = {'C': trial.suggest_float('C', 0.1, 10.0)}
                
                elif self.model_type == "random_forest":
                    p = {'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                         'max_depth': trial.suggest_int('max_depth', 5, 30)}
                
                elif self.model_type == "lightgbm":
                    p = {'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                         'n_estimators': trial.suggest_int('n_estimators', 100, 500)}
                
                elif self.model_type == "linear_svc":
                    p = {'C': trial.suggest_float('C', 0.1, 5.0)}
                
                # Treinamento temporário para avaliação do Trial
                model = self._get_model(p, vectorization_type)
                model.fit(X_train, y_train)
                
                # F1-macro utilizado para lidar com possível desbalanceamento de classes
                return f1_score(y_val, model.predict(X_val), average='macro')

            # Criação do estudo de otimização Bayesiana
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=15)
            self.params = study.best_params
        
        # Ajuste final: Treina o modelo definitivo usando o conjunto completo de treino
        # e os melhores parâmetros encontrados (ou os manuais fornecidos)
        self.best_model = self._get_model(self.params, vectorization_type)
        self.best_model.fit(X_train, y_train)
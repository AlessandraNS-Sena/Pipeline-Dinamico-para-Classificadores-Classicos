from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from .preprocessor import Preprocessor
from .vectorizer import Vectorizer
from .classifier import Classifier

class PipelineEngine:
    """
    Orquestrador principal do pipeline, responsável pela divisão de dados,
    gerenciamento de pré-processamento, vetorização e treinamento.
    """
    def __init__(self, df, text_col, target_col):
        self.df = df
        self.text_col = text_col
        self.target_col = target_col

    def run(self, prep_cfg, vec_cfg, clf_cfg):
        # Divisão estratificada 70/15/15: prioriza manter a proporção das classes
        # em todos os conjuntos para uma avaliação estatística justa.
        try:
            train, temp = train_test_split(
                self.df, 
                test_size=0.3, 
                stratify=self.df[self.target_col], 
                random_state=42
            )
            val, test = train_test_split(
                temp, 
                test_size=0.5, 
                stratify=temp[self.target_col], 
                random_state=42
            )
        except ValueError:
            # Fallback de segurança: caso o dataset possua classes com apenas 1 membro,
            # a estratificação falha. Aqui, garantimos a continuidade do processo sem crash.
            train, temp = train_test_split(
                self.df, 
                test_size=0.3, 
                random_state=42
            )
            val, test = train_test_split(
                temp, 
                test_size=0.5, 
                random_state=42
            )

        # Instanciação modular conforme as configurações recebidas do frontend
        prep = Preprocessor(prep_cfg)
        vec = Vectorizer(vec_cfg['strategy'], vec_cfg['params'])
        
        # Otimização de performance: utiliza processamento em lote via .apply()
        # antes de realizar o fit/transform da vetorização.
        X_train = vec.vectorizer.fit_transform(train[self.text_col].apply(prep.transform))
        X_val = vec.vectorizer.transform(val[self.text_col].apply(prep.transform))
        X_test = vec.vectorizer.transform(test[self.text_col].apply(prep.transform))
        
        # Treinamento do classificador: delega a lógica de busca (Optuna ou Manual) à classe Classifier
        clf = Classifier(clf_cfg['model'], clf_cfg['mode'])
        clf.train(X_train, train[self.target_col], X_val, val[self.target_col])
        
        # Avaliação final estritamente sobre o conjunto de teste (blind test)
        y_pred = clf.best_model.predict(X_test)
        
        # Geração do relatório de métricas formatado para consumo do JSON/Frontend.
        # zero_division=0 evita erros de cálculo caso o modelo ignore classes minoritárias.
        return {
            "metrics": classification_report(test[self.target_col], y_pred, output_dict=True, zero_division=0),
            "best_params": clf.params
        }
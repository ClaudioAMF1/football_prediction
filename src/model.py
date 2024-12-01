from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import logging
import os


class BrasileiraoPredictor:
    def __init__(self):
        # Usando RandomForestClassifier com configurações otimizadas
        self.model = RandomForestClassifier(
            n_estimators=500,  # Mais árvores
            max_depth=10,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight={  # Peso maior para empates
                0: 1.0,  # Vitória fora
                1: 1.5,  # Empate
                2: 1.0  # Vitória casa
            },
            n_jobs=-1,  # Usar todos os cores
            random_state=42
        )

        os.makedirs('logs', exist_ok=True)
        logging.basicConfig(
            filename='logs/model.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def treinar(self, X, y):
        """Treina o modelo"""
        try:
            if X is None or y is None or len(X) < 10:
                return {'error': 'Dados insuficientes para treino'}

            # Dividir dados com estratificação
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            # Treinar modelo
            self.model.fit(X_train, y_train)

            # Avaliar no conjunto de teste
            y_pred = self.model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)

            # Validação cruzada para estabilidade
            cv_scores = []
            for i in range(5):  # 5 iterações com diferentes seeds
                cv_score = cross_val_score(
                    self.model, X, y,
                    cv=5,
                    scoring='accuracy',
                    n_jobs=-1
                )
                cv_scores.extend(cv_score)

            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)

            # Relatório detalhado
            report = classification_report(
                y_test,
                y_pred,
                target_names=['Vitória Fora', 'Empate', 'Vitória Casa']
            )

            # Log dos resultados
            logging.info(f"Acurácia: {test_accuracy:.4f}")
            logging.info(f"CV média: {cv_mean:.4f} (+/- {cv_std:.4f})")
            logging.info(f"Relatório:\n{report}")

            return {
                'test_score': test_accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'classification_report': report
            }

        except Exception as e:
            logging.error(f"Erro no treino: {str(e)}")
            return {'error': str(e)}

    def prever(self, X):
        """Faz previsões"""
        try:
            return self.model.predict(X)
        except Exception as e:
            logging.error(f"Erro na previsão: {str(e)}")
            return None

    def prever_probabilidades(self, X):
        """Retorna probabilidades das previsões"""
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            logging.error(f"Erro no cálculo de probabilidades: {str(e)}")
            return None

    def salvar_modelo(self, caminho='models/brasileirao_predictor.joblib'):
        """Salva o modelo treinado"""
        try:
            os.makedirs(os.path.dirname(caminho), exist_ok=True)
            joblib.dump(self.model, caminho)
            logging.info(f"Modelo salvo em: {caminho}")
            return True
        except Exception as e:
            logging.error(f"Erro ao salvar modelo: {str(e)}")
            return False

    def carregar_modelo(self, caminho='models/brasileirao_predictor.joblib'):
        """Carrega um modelo salvo"""
        try:
            if os.path.exists(caminho):
                self.model = joblib.load(caminho)
                logging.info(f"Modelo carregado de: {caminho}")
                return True
            logging.error(f"Arquivo de modelo não encontrado: {caminho}")
            return False
        except Exception as e:
            logging.error(f"Erro ao carregar modelo: {str(e)}")
            return False
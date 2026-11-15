import numpy as np
import scipy.sparse as sp


class BaseMatrixFactorization:
    """
    Clase base para modelos de Factorización de Matrices.
    Contiene la lógica común de predicción y evaluación (RMSE, Precision@k, NDCG@k).
    """

    def __init__(self, n_factors=20, reg=0.1, n_epochs=15, seed=42):
        self.n_factors = n_factors    # Dimensión latente (k)
        self.reg = reg                # Parámetro de regularización L2 (lambda)
        self.n_epochs = n_epochs      # Número de épocas de entrenamiento
        self.seed = seed              # Semilla para reproducibilidad

        self.P = None                 # Matriz de factores de usuarios (n_users x k)
        self.Q = None                 # Matriz de factores de ítems (n_items x k)
        self.history = {
            'train_rmse': [],
            'val_rmse': [],
            'epoch_time': []
        }

    def _init_matrices(self, n_users, n_items):
        """Inicializa las matrices latentes P y Q con una distribución normal pequeña."""
        np.random.seed(self.seed)
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.history = {'train_rmse': [], 'val_rmse': [], 'epoch_time': []}

    def predict(self, u, i):
        """Predice la valoración del usuario u para el ítem i."""
        return float(np.dot(self.P[u, :], self.Q[i, :]))

    def predict_all(self):
        """Reconstruye la matriz completa aproximada (P * Q^T)."""
        return self.P @ self.Q.T

    def calculate_rmse(self, R):
        """
        Calcula la Raíz del Error Cuadrático Medio (RMSE) sobre las entradas
        observadas (no nulas) de una matriz dispersa R.
        """
        if not sp.issparse(R):
            R = sp.csr_matrix(R)
        users, items = R.nonzero()
        actual = R.data
        predictions = np.sum(self.P[users, :] * self.Q[items, :], axis=1)
        return float(np.sqrt(np.mean((actual - predictions) ** 2)))

    def recommend(self, u, R_train_csr, top_k=10):
        """
        Devuelve los índices de los top_k ítems recomendados para el usuario u,
        excluyendo los ítems que ya ha valorado en el conjunto de entrenamiento.
        """
        scores = self.P[u, :] @ self.Q.T
        seen_items = R_train_csr[u].indices
        scores[seen_items] = -np.inf  # Descartar ítems ya consumidos
        top_items = np.argsort(scores)[::-1][:top_k]
        return top_items

    def evaluate_ranking(self, R_train, R_test, k=10, relevance_threshold=3.5):
        """
        Calcula Precision@k y NDCG@k promediados sobre todos los usuarios con datos en test.
        Un ítem en test se considera relevante si su nota real >= relevance_threshold.
        """
        R_train_csr = R_train.tocsr()
        R_test_csr = R_test.tocsr()
        n_users = R_train_csr.shape[0]

        precisions = []
        ndcgs = []

        for u in range(n_users):
            test_items = R_test_csr[u].indices
            test_ratings = R_test_csr[u].data

            # Filtrar ítems relevantes en test para el usuario u
            relevant_items = set(test_items[test_ratings >= relevance_threshold])
            if len(relevant_items) == 0:
                continue

            # Obtener los top_k recomendados
            recommended = self.recommend(u, R_train_csr, top_k=k)

            # Precision@k
            hits = [1.0 if item in relevant_items else 0.0 for item in recommended]
            precisions.append(np.sum(hits) / k)

            # NDCG@k
            dcg = sum(hit / np.log2(idx + 2) for idx, hit in enumerate(hits))
            idcg_count = min(len(relevant_items), k)
            idcg = sum(1.0 / np.log2(idx + 2) for idx in range(idcg_count))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        return {
            f'Precision@{k}': float(np.mean(precisions)) if precisions else 0.0,
            f'NDCG@{k}': float(np.mean(ndcgs)) if ndcgs else 0.0
        }

import numpy as np
import scipy.sparse as sp


class BaseMatrixFactorization:

    def __init__(self, n_factors=20, reg=0.1, n_epochs=15, seed=42):
        self.n_factors = n_factors
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed

        self.P = None
        self.Q = None
        self.history = {
            'train_rmse': [],
            'val_rmse': [],
            'epoch_time': []
        }

    def _init_matrices(self, n_users, n_items):
        np.random.seed(self.seed)
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.history = {'train_rmse': [], 'val_rmse': [], 'epoch_time': []}

    def predict(self, u, i):
        return float(np.dot(self.P[u, :], self.Q[i, :]))

    def predict_all(self):
        return self.P @ self.Q.T

    def calculate_rmse(self, R):

        R_coo = sp.coo_matrix(R)
        users = R_coo.row
        items = R_coo.col
        actual = R_coo.data
        predictions = np.sum(self.P[users, :] * self.Q[items, :], axis=1)
        return float(np.sqrt(np.mean((actual - predictions) ** 2)))

    def recommend(self, u, R_train_csr, top_k=10):

        if not sp.isspmatrix_csr(R_train_csr):
            R_train_csr = sp.csr_matrix(R_train_csr)
        scores = self.P[u, :] @ self.Q.T
        seen_items = R_train_csr[u].indices
        scores[seen_items] = -np.inf
        top_items = np.argsort(scores)[::-1][:top_k]
        return top_items

    def evaluate_ranking(self, R_train, R_test, k=10, relevance_threshold=3.5):

        R_train_csr = R_train.tocsr()
        R_test_csr = R_test.tocsr()
        n_users = R_train_csr.shape[0]

        precisions = []
        ndcgs = []

        self.calculate_recommendation_quality(R_test_csr, R_train_csr, k, n_users, ndcgs, precisions, relevance_threshold)

        return {
            f'Precision@{k}': float(np.mean(precisions)) if precisions else 0.0,
            f'NDCG@{k}': float(np.mean(ndcgs)) if ndcgs else 0.0
        }

    def calculate_recommendation_quality(self, R_test_csr, R_train_csr, k, n_users, ndcgs,
                                     precisions, relevance_threshold):
        for u in range(n_users):
            test_items = R_test_csr[u].indices
            test_ratings = R_test_csr[u].data

            relevant_items = set(test_items[test_ratings >= relevance_threshold])
            if len(relevant_items) == 0:
                continue

            hits = self.calculate_precision(R_train_csr, k, precisions, relevant_items, u)
            self.calculate_ndcg(hits, k, ndcgs, relevant_items)

    def calculate_precision(self, R_train_csr, k, precisions, relevant_items, u):
        recommended = self.recommend(u, R_train_csr, top_k=k)

        hits = [1.0 if item in relevant_items else 0.0 for item in recommended]
        precisions.append(np.sum(hits) / k)
        return hits

    def calculate_ndcg(self, hits, k, ndcgs, relevant_items):
        dcg = sum(hit / np.log2(idx + 2) for idx, hit in enumerate(hits))
        idcg_count = min(len(relevant_items), k)
        idcg = sum(1.0 / np.log2(idx + 2) for idx in range(idcg_count))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

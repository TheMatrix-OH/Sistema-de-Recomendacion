import time
import numpy as np
import scipy.sparse as sp
from main.BaseMatrixFactorization import BaseMatrixFactorization


class SGD(BaseMatrixFactorization):

    def __init__(self, n_factors=20, lr=0.005, reg=0.1, n_epochs=30, seed=42):
        super().__init__(n_factors=n_factors, reg=reg, n_epochs=n_epochs, seed=seed)
        self.lr = lr

    def fit(self, R_train, R_val=None, verbose=True):
        if not sp.issparse(R_train):
            R_train = sp.csr_matrix(R_train)

        n_users, n_items = R_train.shape
        self._init_matrices(n_users, n_items)

        R_coo = R_train.tocoo()
        users = R_coo.row
        items = R_coo.col
        ratings = R_coo.data
        n_samples = len(ratings)

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.perf_counter()

            indices = np.random.permutation(n_samples)

            for idx in indices:
                u = users[idx]
                i = items[idx]
                r = ratings[idx]

                prediction = np.dot(self.P[u, :], self.Q[i, :])
                err = r - prediction

                p_u_old = self.P[u, :].copy()

                self.P[u, :] += self.lr * (err * self.Q[i, :] - self.reg * self.P[u, :])
                self.Q[i, :] += self.lr * (err * p_u_old - self.reg * self.Q[i, :])

            elapsed = time.perf_counter() - t0
            self.history['epoch_time'].append(elapsed)

            train_rmse = self.calculate_rmse(R_train)
            self.history['train_rmse'].append(train_rmse)

            val_str = ""
            if R_val is not None:
                val_rmse = self.calculate_rmse(R_val)
                self.history['val_rmse'].append(val_rmse)
                val_str = f" | Val RMSE: {val_rmse:.4f}"

            if verbose:
                print(f"[SGD] Época {epoch:2d}/{self.n_epochs} ({elapsed:.2f}s) - "
                      f"Train RMSE: {train_rmse:.4f}{val_str}")

        return self
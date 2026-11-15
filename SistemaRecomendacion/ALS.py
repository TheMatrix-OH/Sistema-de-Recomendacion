import time
import numpy as np
import scipy.sparse as sp
from BaseMatrixFactorization import BaseMatrixFactorization


class ALS(BaseMatrixFactorization):

    def __init__(self, n_factors=20, reg=0.1, n_epochs=15, seed=42):
        super().__init__(n_factors=n_factors, reg=reg, n_epochs=n_epochs, seed=seed)

    def fit(self, R_train, R_val=None, verbose=True):
        if not sp.issparse(R_train):
            R_train = sp.csr_matrix(R_train)

        n_users, n_items = R_train.shape
        self._init_matrices(n_users, n_items)

        R_csr = R_train.tocsr()
        R_csc = R_train.tocsc()

        I_reg = self.reg * np.eye(self.n_factors)

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.perf_counter()

            for u in range(n_users):
                item_indices = R_csr[u].indices
                if len(item_indices) == 0:
                    continue

                ratings_u = R_csr[u].data
                Q_u = self.Q[item_indices, :]

                A_u = Q_u.T @ Q_u + I_reg
                b_u = Q_u.T @ ratings_u
                self.P[u, :] = np.linalg.solve(A_u, b_u)

            for i in range(n_items):
                user_indices = R_csc[:, i].indices
                if len(user_indices) == 0:
                    continue

                ratings_i = R_csc[:, i].data
                P_i = self.P[user_indices, :]

                A_i = P_i.T @ P_i + I_reg
                b_i = P_i.T @ ratings_i
                self.Q[i, :] = np.linalg.solve(A_i, b_i)

            elapsed = time.perf_counter() - t0
            self.history['epoch_time'].append(elapsed)

            train_rmse = self.calculate_rmse(R_csr)
            self.history['train_rmse'].append(train_rmse)

            val_str = ""
            if R_val is not None:
                val_rmse = self.calculate_rmse(R_val)
                self.history['val_rmse'].append(val_rmse)
                val_str = f" | Val RMSE: {val_rmse:.4f}"

            if verbose:
                print(f"[ALS] Época {epoch:2d}/{self.n_epochs} ({elapsed:.2f}s) - "
                      f"Train RMSE: {train_rmse:.4f}{val_str}")

        return self
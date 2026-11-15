import time
import numpy as np
import scipy.sparse as sp
from main.BaseMatrixFactorization import BaseMatrixFactorization


class Baseline(BaseMatrixFactorization):

    def __init__(self, damping=10.0):
        super().__init__(n_factors=1, reg=0.0, n_epochs=1, seed=42)
        self.damping = damping

    def fit(self, R_train, R_val=None, verbose=True):
        t0 = time.perf_counter()

        if not sp.issparse(R_train):
            R_train = sp.csr_matrix(R_train)

        n_users, n_items = R_train.shape
        self.history = {'train_rmse': [], 'val_rmse': [], 'epoch_time': []}

        global_mean = float(np.mean(R_train.data))

        R_csc = R_train.tocsc()
        item_counts = np.diff(R_csc.indptr)
        item_sums = np.array(R_csc.sum(axis=0)).flatten()

        item_means = (item_sums + self.damping * global_mean) / (item_counts + self.damping)

        self.P = np.ones((n_users, 1))
        self.Q = item_means.reshape(-1, 1)

        elapsed = time.perf_counter() - t0
        self.history['epoch_time'].append(elapsed)

        train_rmse = self.calculate_rmse(R_train)
        self.history['train_rmse'].append(train_rmse)

        test_str = ""
        if R_val is not None:
            test_rmse = self.calculate_rmse(R_val)
            self.history['val_rmse'].append(test_rmse)
            test_str = f" | Test RMSE: {test_rmse:.4f}"

        if verbose:
            print(f"[Baseline] Entrenado ({elapsed:.4f}s) - "
                  f"Train RMSE: {train_rmse:.4f}{test_str}")

        return self
import os
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix

N_USERS = 943
N_ITEMS = 1682


def load_ratings_file(filepath, shape=(N_USERS, N_ITEMS)):
    data = np.loadtxt(filepath, delimiter='\t', dtype=np.int64)
    users = data[:, 0] - 1
    items = data[:, 1] - 1
    ratings = data[:, 2].astype(np.float64)

    return sp.csr_matrix((ratings, (users, items)), shape=shape)


def load_train_val_test(data_dir, fold=1, val_ratio=0.1, seed=42):
    np.random.seed(seed)

    R_train, R_val = get_train_and_valid_matrices(data_dir, fold, val_ratio)
    R_test = get_test_matrix(data_dir, fold)

    return R_train, R_val, R_test


def get_train_and_valid_matrices(data_dir, fold, val_ratio):
    base_path = os.path.join(data_dir, f"u{fold}.base")
    base_data = np.loadtxt(base_path, delimiter='\t', dtype=np.int64)

    perm = np.random.permutation(len(base_data))
    n_val = int(len(base_data) * val_ratio)

    return get_train_matrix(base_data, n_val, perm), get_valid_matrix(base_data, n_val, perm)


def get_valid_matrix(base_data: Any, n_val: int, perm: Any) -> csr_matrix[Any]:
    val_data = base_data[perm[:n_val]]
    R_val = sp.csr_matrix(
        (val_data[:, 2].astype(np.float64), (val_data[:, 0] - 1, val_data[:, 1] - 1)),
        shape=(N_USERS, N_ITEMS)
    )
    return R_val


def get_train_matrix(base_data: Any, n_val: int, perm: Any) -> csr_matrix[Any]:
    train_data = base_data[perm[n_val:]]
    R_train = sp.csr_matrix(
        (train_data[:, 2].astype(np.float64), (train_data[:, 0] - 1, train_data[:, 1] - 1)),
        shape=(N_USERS, N_ITEMS)
    )
    return R_train


def get_test_matrix(data_dir, fold: int) -> csr_matrix[Any]:
    test_path = os.path.join(data_dir, f"u{fold}.test")
    R_test = load_ratings_file(test_path, shape=(N_USERS, N_ITEMS))
    return R_test


def load_movie_titles(data_dir):
    item_path = os.path.join(data_dir, "u.item")
    titles = {}
    with open(item_path, mode="r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                movie_id = int(parts[0]) - 1
                titles[movie_id] = parts[1]
    return titles
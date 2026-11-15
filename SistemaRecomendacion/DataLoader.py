import os
import numpy as np
import scipy.sparse as sp

N_USERS = 943
N_ITEMS = 1682


def load_ratings_file(filepath, shape=(N_USERS, N_ITEMS)):
    data = np.loadtxt(filepath, delimiter='\t', dtype=np.int64)
    users = data[:, 0] - 1
    items = data[:, 1] - 1
    ratings = data[:, 2].astype(np.float64)

    return sp.csr_matrix((ratings, (users, items)), shape=shape)


def load_train_val_test(data_dir, fold=1, val_ratio=0.1, seed=42):
    base_path = os.path.join(data_dir, f"u{fold}.base")
    test_path = os.path.join(data_dir, f"u{fold}.test")

    base_data = np.loadtxt(base_path, delimiter='\t', dtype=np.int64)
    np.random.seed(seed)
    perm = np.random.permutation(len(base_data))

    n_val = int(len(base_data) * val_ratio)
    val_data = base_data[perm[:n_val]]
    train_data = base_data[perm[n_val:]]

    R_train = sp.csr_matrix(
        (train_data[:, 2].astype(np.float64), (train_data[:, 0] - 1, train_data[:, 1] - 1)),
        shape=(N_USERS, N_ITEMS)
    )
    R_val = sp.csr_matrix(
        (val_data[:, 2].astype(np.float64), (val_data[:, 0] - 1, val_data[:, 1] - 1)),
        shape=(N_USERS, N_ITEMS)
    )
    R_test = load_ratings_file(test_path, shape=(N_USERS, N_ITEMS))

    return R_train, R_val, R_test


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
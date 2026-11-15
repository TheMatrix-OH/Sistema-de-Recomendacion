import numpy as np
import scipy.sparse as sp
from SGD import SGD
from ALS import ALS

if __name__ == "__main__":
    np.random.seed(42)
    n_users, n_items = 200, 100
    users = np.random.randint(0, n_users, size=2500)
    items = np.random.randint(0, n_items, size=2500)
    ratings = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=2500)

    R_train = sp.csr_matrix((ratings, (users, items)), shape=(n_users, n_items))

    print("--- Entrenando SGD ---")
    sgd_model = SGD(n_factors=10, lr=0.01, reg=0.1, n_epochs=10)
    sgd_model.fit(R_train)

    print("\n--- Entrenando ALS ---")
    als_model = ALS(n_factors=10, reg=0.1, n_epochs=10)
    als_model.fit(R_train)

    print(f"\nPredicción ejemplo (u=0, i=5) -> SGD: {sgd_model.predict(0, 5):.2f} | ALS: {als_model.predict(0, 5):.2f}")
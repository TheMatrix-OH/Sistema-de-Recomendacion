import os
import numpy as np
from DataLoader import load_train_val_test, load_movie_titles
from SGD import SGD
from ALS import ALS
from Baseline import Baseline


def load_data():
    global R_train, R_val, R_test, titles
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "ml-100k")

    print("Cargando datos de MovieLens 100K...")
    R_train, R_val, R_test = load_train_val_test(data_dir, fold=1, val_ratio=0.1, seed=42)
    titles = load_movie_titles(data_dir)

    print(f"Dimensiones matriz: {R_train.shape[0]} usuarios x {R_train.shape[1]} películas")
    print(f"Valoraciones -> Train: {R_train.nnz} | Val: {R_val.nnz} | Test: {R_test.nnz}\n")


def train_sgd():
    print("--- Entrenando SGD ---")
    sgd_model = SGD(n_factors=20, lr=0.005, reg=0.05, n_epochs=30, seed=42)
    sgd_model.fit(R_train, R_val=R_val)
    return sgd_model


def train_als():
    print("\n--- Entrenando ALS ---")
    als_model = ALS(n_factors=20, reg=10.0, n_epochs=30, seed=42)
    als_model.fit(R_train, R_val=R_val)
    return als_model


def train_baseline():
    print("\n--- Entrenando Baseline ---")
    baseline_model = Baseline(damping=10.0)
    baseline_model.fit(R_train, R_val=R_val)
    return baseline_model


def print_predictions_quality(name, model):
    test_rmse = model.calculate_rmse(R_test)
    ranking_metrics = model.evaluate_ranking(R_train, R_test, k=10, relevance_threshold=4.0)
    total_time = np.sum(model.history['epoch_time'])

    print(f"{name:3s} -> Test RMSE: {test_rmse:.4f} | "
          f"Precision@10: {ranking_metrics['Precision@10']:.4f} | "
          f"NDCG@10: {ranking_metrics['NDCG@10']:.4f} | "
          f"Tiempo total: {total_time:.2f}s")

def print_recommendations(name, model, user_idx):
    print(f"\n--- Top 5 Recomendaciones para el Usuario {user_idx + 1} ({name:3s}) ---")
    top_items = model.recommend(user_idx, R_train, top_k=5)
    for rank, item_idx in enumerate(top_items, start=1):
        pred_score = model.predict(user_idx, item_idx)
        print(f"  {rank}. {titles.get(item_idx, 'Desconocida')} (Predicción: {pred_score:.2f} ★)")


if __name__ == "__main__":
    load_data()
    sgd_model = train_sgd()
    als_model = train_als()
    baseline_model = train_baseline()

    print("\n================ RESULTADOS EN TEST ================")
    print_predictions_quality("SGD", sgd_model)
    print_predictions_quality("ALS", als_model)
    print_predictions_quality("Baseline", baseline_model)
    user_idx = 22
    print_recommendations("SGD", sgd_model, user_idx)
    print_recommendations("ALS", als_model, user_idx)
    print_recommendations("Baseline", baseline_model, user_idx)

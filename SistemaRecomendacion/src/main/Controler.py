import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from main.DataLoader import load_train_val_test, load_movie_titles
from main.SGD import SGD
from main.ALS import ALS
from main.Baseline import Baseline


def load_data():
    global R_train, R_val, R_test, titles
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, "resources", "ml-100k")

    print("Cargando datos de MovieLens 100K...")
    R_train, R_val, R_test = load_train_val_test(data_dir, fold=1, val_ratio=0.1, seed=42)
    titles = load_movie_titles(data_dir)

    print(f"Dimensiones matriz: {R_train.shape[0]} usuarios x {R_train.shape[1]} películas")
    print(f"Valoraciones -> Train: {R_train.nnz} | Val: {R_val.nnz} | Test: {R_test.nnz}\n")


def train_sgd():
    print("--- Entrenando SGD ---")
    sgd_model = SGD(n_factors=20, lr=0.005, reg=0.05, n_epochs=30, seed=42)
    sgd_model.fit(R_train, R_val=R_test)
    return sgd_model


def train_als():
    print("\n--- Entrenando ALS ---")
    als_model = ALS(n_factors=20, reg=10.0, n_epochs=30, seed=42)
    als_model.fit(R_train, R_val=R_test)
    return als_model


def train_baseline():
    print("\n--- Entrenando Baseline ---")
    baseline_model = Baseline(damping=10.0)
    baseline_model.fit(R_train, R_val=R_test)
    return baseline_model


def print_predictions_quality(name, model):
    test_rmse = model.calculate_rmse(R_test)
    ranking_metrics = model.evaluate_ranking(R_train, R_test, k=10, relevance_threshold=4.0)
    total_time = np.sum(model.history['epoch_time'])

    print(f"{name:8s} -> Test RMSE: {test_rmse:.4f} | "
          f"Precision@10: {ranking_metrics['Precision@10']:.4f} | "
          f"NDCG@10: {ranking_metrics['NDCG@10']:.4f} | "
          f"Tiempo total: {total_time:.2f}s")


def print_recommendations(name, model, user_idx):
    print(f"\n--- Top 5 Recomendaciones para el Usuario {user_idx + 1} ({name:3s}) ---")
    top_items = model.recommend(user_idx, R_train, top_k=5)
    for rank, item_idx in enumerate(top_items, start=1):
        pred_score = model.predict(user_idx, item_idx)
        print(f"  {rank}. {titles.get(item_idx, 'Desconocida')} (Predicción: {pred_score:.2f} ★)")


def save_and_plot_convergence(sgd_model, als_model, baseline_model, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    n_epochs = len(sgd_model.history['train_rmse'])
    base_train = baseline_model.history['train_rmse'][0]
    base_test = baseline_model.history['val_rmse'][0]
    base_time = baseline_model.history['epoch_time'][0]

    for epoch in range(1, n_epochs + 1):
        idx = epoch - 1
        rows.append({
            "method": "SGD",
            "epoch": epoch,
            "train_rmse": sgd_model.history['train_rmse'][idx],
            "test_rmse": sgd_model.history['val_rmse'][idx],
            "epoch_time": sgd_model.history['epoch_time'][idx]
        })
        rows.append({
            "method": "ALS",
            "epoch": epoch,
            "train_rmse": als_model.history['train_rmse'][idx],
            "test_rmse": als_model.history['val_rmse'][idx],
            "epoch_time": als_model.history['epoch_time'][idx]
        })
        rows.append({
            "method": "Baseline",
            "epoch": epoch,
            "train_rmse": base_train,
            "test_rmse": base_test,
            "epoch_time": base_time
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "convergence_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nDatos de convergencia guardados en: {csv_path}")

    plt.figure(figsize=(10, 6))

    for method, color in [("SGD", "tab:blue"), ("ALS", "tab:orange"), ("Baseline", "tab:green")]:
        data = df[df["method"] == method].sort_values("epoch")
        plt.plot(data["epoch"], data["test_rmse"], marker="o", markersize=4,
                 linewidth=2, color=color, label=f"{method} (Test RMSE)")
        plt.plot(data["epoch"], data["train_rmse"], linestyle="--",
                 linewidth=1.5, color=color, alpha=0.7, label=f"{method} (Train RMSE)")

    plt.xlabel("Época")
    plt.ylabel("RMSE")
    plt.title("Convergencia por época: Train RMSE vs Test RMSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "convergence_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"Gráfica de convergencia guardada en: {plot_path}")
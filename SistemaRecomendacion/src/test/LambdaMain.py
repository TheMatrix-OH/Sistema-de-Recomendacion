import os

from SistemaRecomendacion.src.main.DataLoader import load_train_val_test, load_movie_titles
from SistemaRecomendacion.src.test.LambdaRunner import LambdaTester


def load_data():
    global R_train, R_val, R_test, titles
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, r"resources\ml-100k")

    print("Cargando datos de MovieLens 100K...")
    R_train, R_val, R_test = load_train_val_test(data_dir, fold=1, val_ratio=0.1, seed=42)
    titles = load_movie_titles(data_dir)

    print(f"Dimensiones matriz: {R_train.shape[0]} usuarios x {R_train.shape[1]} películas")
    print(f"Valoraciones -> Train: {R_train.nnz} | Val: {R_val.nnz} | Test: {R_test.nnz}\n")


if __name__ == "__main__":
    load_data()

    lambda_values = [0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    seeds = [21, 42, 123]

    experiment = LambdaTester(
        lambda_values=lambda_values,
        seeds=seeds,
        k=20,
        n_epochs=30
    )

    experiment.run(R_train, R_val, R_test)
    experiment.plot()
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SistemaRecomendacion.src.main.SGD import SGD
from SistemaRecomendacion.src.main.ALS import ALS
from SistemaRecomendacion.src.main.Baseline import Baseline


class LambdaTester:

    def __init__(
        self,
        lambda_values,
        seeds,
        k=20,
        n_epochs=30,
        output_dir="results",
        sgd_lr=0.005,
        damping=10.0
    ):
        self.lambda_values = lambda_values
        self.seeds = seeds
        self.k = k
        self.n_epochs = n_epochs
        self.output_dir = output_dir

        self.sgd_lr = sgd_lr
        self.damping = damping

        os.makedirs(self.output_dir, exist_ok=True)

        self.results = []
        self.summary = None

    def run(self, R_train, R_val, R_test):
        self.results = []

        print("\n========== BASELINE ==========")

        baseline = Baseline(damping=self.damping)

        start = time.perf_counter()
        baseline.fit(R_train, R_val=R_val, verbose=False)
        elapsed = time.perf_counter() - start

        baseline_rmse = baseline.calculate_rmse(R_test)

        print(f"Baseline -> Test RMSE: {baseline_rmse:.4f}")

        self.results.append({
            "method": "Baseline",
            "lambda": np.nan,
            "seed": np.nan,
            "test_rmse": baseline_rmse,
            "time": elapsed
        })

        for reg in self.lambda_values:

            print(f"\n========== lambda = {reg} ==========")

            for seed in self.seeds:

                print(f"\nSGD | k={self.k} | lambda={reg} | seed={seed}")

                sgd_model = SGD(
                    n_factors=self.k,
                    lr=self.sgd_lr,
                    reg=reg,
                    n_epochs=self.n_epochs,
                    seed=seed
                )

                start = time.perf_counter()
                sgd_model.fit(R_train, R_val=R_val, verbose=False)
                elapsed = time.perf_counter() - start

                test_rmse = sgd_model.calculate_rmse(R_test)

                self.results.append({
                    "method": "SGD",
                    "lambda": reg,
                    "seed": seed,
                    "test_rmse": test_rmse,
                    "time": elapsed
                })

                print(f"Test RMSE: {test_rmse:.4f}")

                print(f"\nALS | k={self.k} | lambda={reg} | seed={seed}")

                als_model = ALS(
                    n_factors=self.k,
                    reg=reg,
                    n_epochs=self.n_epochs,
                    seed=seed
                )

                start = time.perf_counter()
                als_model.fit(R_train, R_val=R_val, verbose=False)
                elapsed = time.perf_counter() - start

                test_rmse = als_model.calculate_rmse(R_test)

                self.results.append({
                    "method": "ALS",
                    "lambda": reg,
                    "seed": seed,
                    "test_rmse": test_rmse,
                    "time": elapsed
                })

                print(f"Test RMSE: {test_rmse:.4f}")

        results_df = pd.DataFrame(self.results)

        results_path = os.path.join(
            self.output_dir,
            "experiment_lambda_results.csv"
        )

        results_df.to_csv(results_path, index=False)

        print(f"\nResultados individuales guardados en: {results_path}")

        summary = (
            results_df[results_df["method"] != "Baseline"]
            .groupby(["method", "lambda"])["test_rmse"]
            .agg(["mean", "std"])
            .reset_index()
        )

        summary.rename(columns={
            "mean": "mean_rmse",
            "std": "std_rmse"
        }, inplace=True)

        summary["std_rmse"] = summary["std_rmse"].fillna(0.0)

        baseline_row = results_df[
            results_df["method"] == "Baseline"
        ].iloc[0]

        baseline_summary = pd.DataFrame([
            {
                "method": "Baseline",
                "lambda": reg,
                "mean_rmse": baseline_row["test_rmse"],
                "std_rmse": 0.0
            }
            for reg in self.lambda_values
        ])

        self.summary = pd.concat(
            [summary, baseline_summary],
            ignore_index=True
        )

        self.summary.sort_values(
            ["method", "lambda"],
            inplace=True
        )

        summary_path = os.path.join(
            self.output_dir,
            "experiment_lambda_summary.csv"
        )

        self.summary.to_csv(summary_path, index=False)

        print(f"Resumen guardado en: {summary_path}")

        print("\n========== RESUMEN ==========")
        print(self.summary.to_string(index=False))

        return self.summary

    def plot(self):
        if self.summary is None:
            raise RuntimeError("Primero debes ejecutar run().")

        plt.figure(figsize=(9, 6))

        for method in ["SGD", "ALS", "Baseline"]:

            data = self.summary[
                self.summary["method"] == method
            ].sort_values("lambda")

            plt.errorbar(
                data["lambda"],
                data["mean_rmse"],
                yerr=data["std_rmse"],
                marker="o",
                capsize=4,
                linewidth=2,
                label=method
            )

        plt.xlabel("Parámetro de regularización (λ)")
        plt.ylabel("Test RMSE")
        plt.title(f"RMSE según la regularización λ (k={self.k})")
        plt.xticks(self.lambda_values)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(
            self.output_dir,
            "experiment_lambda_plot.png"
        )

        plt.savefig(plot_path, dpi=300)
        plt.show()

        print(f"Gráfica guardada en: {plot_path}")
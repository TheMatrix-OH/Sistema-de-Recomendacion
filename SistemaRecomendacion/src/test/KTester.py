import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SistemaRecomendacion.src.main.SGD import SGD
from SistemaRecomendacion.src.main.ALS import ALS
from SistemaRecomendacion.src.main.Baseline import Baseline


class KTester:

    def __init__(
        self,
        k_values,
        seeds,
        n_epochs=30,
        output_dir="results",
        sgd_lr=0.005,
        sgd_reg=0.05,
        als_reg=10.0,
        damping=10.0
    ):
        self.k_values = k_values
        self.seeds = seeds
        self.n_epochs = n_epochs
        self.output_dir = output_dir

        self.sgd_lr = sgd_lr
        self.sgd_reg = sgd_reg
        self.als_reg = als_reg
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
            "k": np.nan,
            "seed": np.nan,
            "test_rmse": baseline_rmse,
            "time": elapsed
        })

        for k in self.k_values:

            print(f"\n========== k = {k} ==========")

            for seed in self.seeds:

                print(f"\nSGD | k={k} | seed={seed}")

                sgd_model = SGD(
                    n_factors=k,
                    lr=self.sgd_lr,
                    reg=self.sgd_reg,
                    n_epochs=self.n_epochs,
                    seed=seed
                )

                start = time.perf_counter()
                sgd_model.fit(R_train, R_val=R_val, verbose=False)
                elapsed = time.perf_counter() - start

                test_rmse = sgd_model.calculate_rmse(R_test)

                self.results.append({
                    "method": "SGD",
                    "k": k,
                    "seed": seed,
                    "test_rmse": test_rmse,
                    "time": elapsed
                })

                print(f"Test RMSE: {test_rmse:.4f}")

                print(f"\nALS | k={k} | seed={seed}")

                als_model = ALS(
                    n_factors=k,
                    reg=self.als_reg,
                    n_epochs=self.n_epochs,
                    seed=seed
                )

                start = time.perf_counter()
                als_model.fit(R_train, R_val=R_val, verbose=False)
                elapsed = time.perf_counter() - start

                test_rmse = als_model.calculate_rmse(R_test)

                self.results.append({
                    "method": "ALS",
                    "k": k,
                    "seed": seed,
                    "test_rmse": test_rmse,
                    "time": elapsed
                })

                print(f"Test RMSE: {test_rmse:.4f}")

        results_df = pd.DataFrame(self.results)

        results_path = os.path.join(
            self.output_dir,
            "experiment_a_results.csv"
        )

        results_df.to_csv(results_path, index=False)

        print(f"\nResultados individuales guardados en: {results_path}")

        summary = (
            results_df[results_df["method"] != "Baseline"]
            .groupby(["method", "k"])["test_rmse"]
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
                "k": k,
                "mean_rmse": baseline_row["test_rmse"],
                "std_rmse": 0.0
            }
            for k in self.k_values
        ])

        self.summary = pd.concat(
            [summary, baseline_summary],
            ignore_index=True
        )

        self.summary.sort_values(
            ["method", "k"],
            inplace=True
        )

        summary_path = os.path.join(
            self.output_dir,
            "experiment_a_summary.csv"
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
            ].sort_values("k")

            plt.errorbar(
                data["k"],
                data["mean_rmse"],
                yerr=data["std_rmse"],
                marker="o",
                capsize=4,
                linewidth=2,
                label=method
            )

        plt.xlabel("Número de factores latentes (k)")
        plt.ylabel("Test RMSE")
        plt.title("Experiment A: RMSE según el número de factores")
        plt.xticks(self.k_values)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(
            self.output_dir,
            "experiment_a_plot.png"
        )

        plt.savefig(plot_path, dpi=300)
        plt.show()

        print(f"Gráfica guardada en: {plot_path}")

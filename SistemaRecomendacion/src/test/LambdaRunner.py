import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

from main.SGD import SGD
from main.ALS import ALS
from main.Baseline import Baseline


class LambdaRunner:

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

        self.test_baseline(R_test, R_train, R_val)

        for lambda_value in self.lambda_values:
            print(f"\n========== lambda = {lambda_value} ==========")
            for seed in self.seeds:
                self.test_sgd(R_test, R_train, R_val, lambda_value, seed)
                self.test_als(R_test, R_train, R_val, lambda_value, seed)

        results_df = self.save_results()

        self.make_summary(results_df)
        self.save_summary()
        self.show_summary()

        return self.summary


    def test_baseline(self, R_test, R_train, R_val):
        print("\n========== BASELINE ==========")

        baseline = Baseline(damping=self.damping)

        start = time.perf_counter()
        baseline.fit(R_train, R_val, False)
        elapsed = time.perf_counter() - start

        baseline_rmse = baseline.calculate_rmse(R_test)
        print(f"Baseline -> Test RMSE: {baseline_rmse:.4f}")

        self.append_result("Baseline", elapsed, np.nan, np.nan, baseline_rmse)


    def test_sgd(self, R_test, R_train, R_val, lambda_value, seed):
        print(f"\nSGD | lambda={lambda_value} | seed={seed}")

        sgd_model = SGD(
            n_factors=self.k,
            lr=self.sgd_lr,
            reg=lambda_value,
            n_epochs=self.n_epochs,
            seed=seed
        )

        start = time.perf_counter()
        sgd_model.fit(R_train, R_val, False)
        elapsed = time.perf_counter() - start

        test_rmse = sgd_model.calculate_rmse(R_test)
        print(f"Test RMSE: {test_rmse:.4f}")

        self.append_result("SGD", elapsed, lambda_value, seed, test_rmse)


    def test_als(self, R_test, R_train, R_val, lambda_value, seed):
        print(f"\nALS | lambda={lambda_value} | seed={seed}")

        als_model = ALS(
            n_factors=self.k,
            reg=lambda_value,
            n_epochs=self.n_epochs,
            seed=seed
        )

        start = time.perf_counter()
        als_model.fit(R_train, R_val, False)
        elapsed = time.perf_counter() - start

        test_rmse = als_model.calculate_rmse(R_test)

        print(f"Test RMSE: {test_rmse:.4f}")

        self.append_result("ALS", elapsed, lambda_value, seed, test_rmse)


    def append_result(self, method, elapsed, lambda_value, seed, test_rmse):
        self.results.append({
            "method": method,
            "lambda": lambda_value,
            "seed": seed,
            "test_rmse": test_rmse,
            "time": elapsed
        })


    def save_results(self):
        results_df = pd.DataFrame(self.results)
        results_path = os.path.join(
            self.output_dir,
            "lambda_results.csv"
        )

        results_df.to_csv(results_path, index=False)
        print(f"\nResultados individuales guardados en: {results_path}")

        return results_df


    def make_summary(self, results_df: DataFrame):
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
                "lambda": lambda_value,
                "mean_rmse": baseline_row["test_rmse"],
                "std_rmse": 0.0
            }
            for lambda_value in self.lambda_values
        ])

        self.summary = pd.concat(
            [summary, baseline_summary],
            ignore_index=True
        )

        self.summary.sort_values(
            ["method", "lambda"],
            inplace=True
        )


    def save_summary(self):
        summary_path = os.path.join(
            self.output_dir,
            "lambda_summary.csv"
        )

        self.summary.to_csv(summary_path, index=False)

        print(f"Resumen guardado en: {summary_path}")


    def show_summary(self):
        print("\n========== RESUMEN ==========")
        print(self.summary.to_string(index=False))


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

        plt.xlabel("Parámetro de regularización (lambda)")
        plt.ylabel("Test RMSE")
        plt.title("Lambda Runner: RMSE según la regularización")
        plt.xticks(self.lambda_values)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(
            self.output_dir,
            "lambda_plot.png"
        )

        plt.savefig(plot_path, dpi=300)
        plt.show()

        print(f"Gráfica guardada en: {plot_path}")

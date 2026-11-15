from SistemaRecomendacion.src.main.Controler import *


if __name__ == "__main__":
    load_data()
    sgd_model = train_sgd()
    als_model = train_als()
    baseline_model = train_baseline()

    print("\n================ RESULTADOS EN TEST ================")
    print_predictions_quality("SGD", sgd_model)
    print_predictions_quality("ALS", als_model)
    print_predictions_quality("Baseline", baseline_model)
    user_idx = 2
    print_recommendations("SGD", sgd_model, user_idx)
    print_recommendations("ALS", als_model, user_idx)
    print_recommendations("Baseline", baseline_model, user_idx)

    save_and_plot_convergence(sgd_model, als_model, baseline_model)
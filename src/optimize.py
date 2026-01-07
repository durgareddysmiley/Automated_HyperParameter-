import optuna
import mlflow
import mlflow.xgboost
import numpy as np
import random
import time
import json
import os

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances
)

from data_loader import load_and_split_data
from objective import objective


def main():
    start_time = time.time()

    np.random.seed(42)
    random.seed(42)

    # 🔥 ABSOLUTE PATH (IMPORTANT)
    OUTPUT_DIR = "/app/outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    STUDY_DB = f"{OUTPUT_DIR}/optuna_study.db"

    X_train, X_test, y_train, y_test = load_and_split_data()

    mlflow.set_tracking_uri("file:///app/outputs/mlruns")
    mlflow.set_experiment("optuna-xgboost-optimization")

    study = optuna.create_study(
        study_name="xgboost-housing-optimization",
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        storage=f"sqlite:///{STUDY_DB}",
        load_if_exists=True
    )

    def wrapped_objective(trial):
        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            value = objective(trial, X_train, y_train)

            mlflow.log_params(trial.params)
            mlflow.log_metric("cv_mse", -value)
            mlflow.log_metric("cv_rmse", np.sqrt(-value))
            mlflow.log_metric("trial_number", trial.number)
            mlflow.set_tag("trial_state", trial.state.name)

            return value

    study.optimize(wrapped_objective, n_trials=100, n_jobs=2)

    best_params = study.best_params

    model = XGBRegressor(
        **best_params,
        random_state=42,
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    test_mse = mean_squared_error(y_test, preds)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, preds)

    with mlflow.start_run(run_name="best_model"):
        mlflow.set_tag("best_model", "true")
        mlflow.log_params(best_params)
        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)
        mlflow.xgboost.log_model(model, artifact_path="model")

        fig1 = plot_optimization_history(study)
        fig1.savefig("/app/outputs/optimization_history.png", bbox_inches="tight")
        mlflow.log_artifact("/app/outputs/optimization_history.png")

        fig2 = plot_param_importances(study)
        fig2.savefig("/app/outputs/param_importance.png", bbox_inches="tight")
        mlflow.log_artifact("/app/outputs/param_importance.png")

    results = {
        "n_trials_completed": len([t for t in study.trials if t.state.name == "COMPLETE"]),
        "n_trials_pruned": len([t for t in study.trials if t.state.name == "PRUNED"]),
        "best_cv_rmse": np.sqrt(-study.best_value),
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "best_params": best_params,
        "optimization_time_seconds": time.time() - start_time
    }

    with open("/app/outputs/results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()

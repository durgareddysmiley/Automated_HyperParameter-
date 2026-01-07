import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import optuna
import mlflow
import mlflow.xgboost
import numpy as np
import random
import time
import json
import os
import shutil

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
import evaluate


def main():
    start_time = time.time()

    np.random.seed(42)
    random.seed(42)

    # 🔥 ABSOLUTE PATH (IMPORTANT)
    OUTPUT_DIR = "/app/outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 🔥 INTERNAL DB to avoid locking on Windows mounts
    INTERNAL_DB = "temp_optuna_study.db"
    STUDY_DB_PATH = f"sqlite:///{INTERNAL_DB}"

    X_train, X_test, y_train, y_test = load_and_split_data()

    mlflow.set_tracking_uri("file:///app/outputs/mlruns")
    mlflow.set_experiment("optuna-xgboost-optimization")

    study = optuna.create_study(
        study_name="xgboost-housing-optimization",
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        storage=STUDY_DB_PATH,
        load_if_exists=True
    )

    def wrapped_objective(trial):
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            try:
                value = objective(trial, X_train, y_train)
                
                mlflow.log_params(trial.params)
                mlflow.log_metric("cv_mse", -value)
                mlflow.log_metric("cv_rmse", np.sqrt(-value))
                mlflow.log_metric("trial_number", trial.number)
                mlflow.set_tag("trial_state", "COMPLETE")
                
                return value
            except optuna.exceptions.TrialPruned:
                mlflow.set_tag("trial_state", "PRUNED")
                raise
            except Exception as e:
                mlflow.set_tag("trial_state", "FAIL")
                mlflow.log_param("error", str(e))
                raise e

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

        ax1 = plot_optimization_history(study)
        ax1.figure.savefig("/app/outputs/optimization_history.png", bbox_inches="tight")
        mlflow.log_artifact("/app/outputs/optimization_history.png")

        ax2 = plot_param_importances(study)
        ax2.figure.savefig("/app/outputs/param_importance.png", bbox_inches="tight")
        mlflow.log_artifact("/app/outputs/param_importance.png")

    # 10. Generate Results JSON (Using evaluate.py)
    total_time = time.time() - start_time
    evaluate.generate_results(study, test_rmse, test_r2, total_time, OUTPUT_DIR)

    # 11. Copy DB to outputs for persistence
    print("💾 Saving Optuna database...")
    shutil.copy(INTERNAL_DB, f"{OUTPUT_DIR}/optuna_study.db")


if __name__ == "__main__":
    main()

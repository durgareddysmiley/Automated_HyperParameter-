# Optuna + MLflow Hyperparameter Optimization Pipeline

## Project Overview
This project implements a production-grade, automated hyperparameter optimization pipeline for the California Housing dataset using XGBoost. It demonstrates advanced MLOps practices by combining **Optuna** for efficient Bayesian optimization (TPE) and Pruning with **MLflow** for comprehensive experiment tracking and reproducibility.

The pipeline is fully containerized using Docker, ensuring that the optimization process is reproducible and environment-agnostic.

### Key Features
- **Automated Hyperparameter Tuning**: searches a 7-dimensional space using Tree-structured Parzen Estimator (TPE).
- **Intelligent Pruning**: Uses `MedianPruner` to stop unpromising trials early, saving computational resources.
- **Experiment Tracking**: Logs every trial's parameters, metrics (MSE, RMSE, R²), and artifacts to MLflow.
- **Parallel Execution**: Configured to run trials in parallel (`n_jobs=2`) using a SQLite backend for state sharing.
- **Reproducibility**: Strict random seed management across all components (Optuna, XGBoost, Numpy/Random split).

## Docker Setup and Execution

The entire pipeline is packaged in a Docker container.

### Prerequisites
- Docker installed and running.

### Build the Image
Build the Docker image using the provided `Dockerfile`.
```bash
docker build -t optuna-mlflow-pipeline .
```

### Run the Pipeline
Run the container, mounting a local `outputs` directory to capture the results.
```bash
# On Linux/Mac
docker run -v $(pwd)/outputs:/app/outputs optuna-mlflow-pipeline

# On Windows (PowerShell)
docker run -v ${PWD}/outputs:/app/outputs optuna-mlflow-pipeline
```
*Note: The pipeline will run for 100 trials and typically completes in under 15 minutes depending on hardware.*

## Expected Outputs

After execution, the mounted `outputs/` directory will contain:

1.  **`results.json`**: A summary JSON file containing:
    -   `best_params`: The optimal hyperparameters found.
    -   `test_rmse` & `test_r2`: Performance of the best model on the hold-out test set.
    -   `optimization_time_seconds`: Total runtime.
2.  **`mlruns/`**: The local MLflow tracking store (file-based).
3.  **`optuna_study.db`**: The SQLite database containing the full study state.
4.  **`optimization_history.png`**: Plot showing objective value improvement over time.
5.  **`param_importance.png`**: Visualization of which hyperparameters impacted the model most.

## Optimization Strategy

### Search Space
We optimize 7 hyperparameters:
- `n_estimators`: [50, 300]
- `max_depth`: [3, 10]
- `learning_rate`: [0.001, 0.3] (log scale)
- `subsample`: [0.6, 1.0]
- `colsample_bytree`: [0.6, 1.0]
- `min_child_weight`: [1, 10]
- `gamma`: [0, 0.5]

### Objective
The pipeline minimizes the **Negative Mean Squared Error (MSE)** via 5-fold Cross-Validation.

## Analysis and Results
An analysis notebook is provided in `notebooks/analysis.ipynb` to explore the results interactively.

**Typical Outcomes:**
- **Baseline RMSE**: ~0.50 - 0.55
- **Tuned RMSE**: < 0.50 (often ~0.46-0.48)
- **R² Score**: > 0.80

The most influential parameters are typically `learning_rate` and `max_depth`, followed by `min_child_weight`.

## Improvements & Best Practices
- **Robust Parallelism**: The optimization loop is refactored to support multiprocessing (`n_jobs > 1`) safely by handling pickling requirements.
- **Error Handling**: Trials are tagged as `FAIL` in MLflow if they crash, ensuring the study doesn't abort entirely.
- **Clean Artifact Management**: We deliberately separate the "Optimization" phase from the final "Best Model" training to ensure the final model is trained on the full training set and evaluated on a pure test set.

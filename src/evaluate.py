import json
import numpy as np
import optuna
from pathlib import Path

def generate_results(study, test_rmse, test_r2, optimization_time, output_dir):
    """
    Generates the results.json file summarizing the optimization.
    """
    
    # Calculate stats
    n_trials_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_trials_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    
    # Best CV RMSE (Note: optimization was maximizing negative MSE)
    best_cv_rmse = np.sqrt(-study.best_value)
    
    results = {
        "n_trials_completed": int(n_trials_completed),
        "n_trials_pruned": int(n_trials_pruned),
        "best_cv_rmse": float(best_cv_rmse),
        "test_rmse": float(test_rmse),
        "test_r2": float(test_r2),
        "best_params": study.best_params,
        "optimization_time_seconds": float(optimization_time)
    }
    
    output_path = Path(output_dir) / "results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Results saved to {output_path}")

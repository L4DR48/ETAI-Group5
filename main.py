"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full (deliberately simple) pipeline:
    load config -> load data -> preprocess -> split -> train
    -> evaluate (train & test) -> save results
"""

import os
import joblib
import shutil
import tempfile
import optuna
import yaml
import sys
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


from src.tuning import tune_pipeline, nested_cross_validate, tuning_report

from src.data import load_data
from src.preprocessing import clean_dataset, drop_duplicate_rows, split_features_target, split_train_test
from src.model import build_pipeline
from src.evaluate import (
    build_scorers, cross_validate_pipeline, cv_report, oof_classification_report, fairness_report,
)
from src.results import save_run, save_scores


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    config = load_config(config_path)

    run = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_type = config["model"]["type"]

    df = load_data(config["data"]["path"])
    df_clean = clean_dataset(df, config["diagnostics"])
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    X_train, X_test, y_train, y_test, extras_train, extras_test = split_train_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    eval_config = config["cv"]
    eval_type = eval_config.get("type", "kfold")
    shuffle = eval_config.get("shuffle", True)

    if eval_type == "holdout":
        cv = StratifiedShuffleSplit(
            n_splits=1,
            test_size=eval_config.get("val_size", 0.2),
            random_state=eval_config.get("random_state", 42),
        )
    else:
        cv = StratifiedKFold(
            n_splits=eval_config["n_splits"], 
            shuffle=shuffle,
            random_state=eval_config.get("random_state") if shuffle else None
        )
    
    metrics = eval_config["metrics"]
    selection_metric = eval_config["selection_metric"]
    scorers = build_scorers(metrics)
    n_jobs=eval_config.get("n_jobs", 1)
    selection_config = config.get("feature_selection") or {}
    selection_enabled = selection_config.get("enabled", False)

    print(f"Model: {model_type}   |   selection metric: {selection_metric}   |   "
          f"{eval_config['n_splits']}-fold CV   |   feature selection: "
          f"{'min_votes ' + str(selection_config.get('min_votes', 2)) if selection_enabled else 'off'}")
    
    print(f"Data:  {len(X_train)} development rows   |   {len(X_test)} test rows locked away (not used)")

    cache_dir = tempfile.mkdtemp(prefix="pipeline_cache_")

    pipeline = build_pipeline(config["preprocessing"], config["model"], selection_config, memory=cache_dir)
    floor_pipeline = build_pipeline(config["preprocessing"], {"type": "dummy"})
    floor_scores, _ = cross_validate_pipeline(floor_pipeline, X_train, y_train, cv,
                                              {selection_metric: scorers[selection_metric]}, selection_metric, n_jobs)
    fold_scores, oof_proba = cross_validate_pipeline(pipeline, X_train, y_train, cv, scorers, selection_metric, n_jobs)
    floor, untuned = floor_scores["validation"].mean(), fold_scores["validation"].mean()

    print("\nDefault hyperparameters (config.yaml)")
    print(f"  {selection_metric}: {untuned:.3f} ± {fold_scores['validation'].std(ddof=1):.3f}   "
          f"train-validation gap: {fold_scores['gap'].mean():+.3f}   no-model floor: {floor:.3f}")
    report = (f"Hyperparameters from config.yaml: {config['model'].get('params')}\n\n"
              + cv_report(fold_scores, metrics, selection_metric, "Cross-validation")
              + f"\n\nNo-model floor (dummy, same folds): {selection_metric} = {floor:.3f}")
    final_scores = fold_scores

    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)

    if tuning_enabled:
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(f"tuning is enabled but config.yaml has no tuning.search_spaces for "
                             f"'{model_type}'. Add one, or set tuning.enabled: false.")
        search_space = dict(search_spaces[model_type])
        if selection_enabled:
            search_space.update(selection_config.get("search_space") or {})
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)

        # Honest estimate on the same outer folds; its out-of-fold probabilities feed the reports.
        print(f"\nTuning ({n_trials} trials, nested CV) ... this takes a while", flush=True)
        final_scores, oof_proba = nested_cross_validate(
            pipeline, X_train, y_train, cv, inner_cv, scorers, selection_metric,
            search_space, n_trials, tuning_seed, n_jobs=n_jobs,
        )
        print(f"  {selection_metric}: {final_scores['validation'].mean():.3f} ± "
              f"{final_scores['validation'].std(ddof=1):.3f}   (honest estimate of the tuned model)")
        report += "\n\n" + cv_report(final_scores, metrics, selection_metric,
                                     "Nested cross-validation (the tuning procedure, estimated honestly)")

        # The hyperparameters that are kept: the same procedure, once, on all development rows.
        pipeline, study = tune_pipeline(pipeline, X_train, y_train, inner_cv, scorers[selection_metric],
                                        search_space, n_trials, tuning_seed, n_jobs=n_jobs)
        print("  Chosen hyperparameters:")
        for name, value in study.best_params.items():
            print(f"    {name.replace('model__', ''):<20s} {value:.4g}" if isinstance(value, float)
                  else f"    {name.replace('model__', ''):<20s} {value}")
        report += "\n\n" + tuning_report(study, final_scores, selection_metric)

    print(f"\nAll metrics ({'tuned model' if tuning_enabled else 'default hyperparameters'}, mean ± std)")
    for name in metrics:
        print(f"  {name:<18s} {final_scores[name].mean():.3f} ± {final_scores[name].std(ddof=1):.3f}")
    report += "\n\n" + oof_classification_report(y_train, oof_proba)
    report += "\n" + fairness_report(y_train, oof_proba, extras_train, sensitive_attr=config["data"]["sensitive_attr"])

    pipeline.fit(X_train, y_train)
    refit = f"Final model: {model_type} refit on all {len(X_train)} development rows."
    if selection_enabled:
        all_features = pipeline.named_steps["prep"].get_feature_names_out()
        kept = pipeline.named_steps["select"].get_feature_names_out(all_features)
        refit += f"\nFeatures kept: {len(kept)} of {len(all_features)}: {', '.join(kept)}"
    report += "\n" + refit + "\n"

    row = {"run": run, "data": config["data"]["path"], "model": model_type, "tuned": tuning_enabled,
           "encoder": config["preprocessing"]["encoder"], "scaler": config["preprocessing"]["scaler"],
           "imputer": config["preprocessing"].get("imputation", {}).get("numeric_strategy", "median"),
           "feature_selection": (f"min_votes {pipeline.named_steps['select'].min_votes}" if selection_enabled else "off"),
           "params": str(config["model"].get("params")), "tuned_params": str(study.best_params) if tuning_enabled else "",
           "metric": selection_metric, "mean": final_scores["validation"].mean(),
           "std": final_scores["validation"].std(ddof=1), "untuned": untuned, "floor": floor,
           **{f"fold_{i}": score for i, score in enumerate(final_scores["validation"], start=1)}}
    runs = save_scores(config.get("output", {}).get("scoreboard", "results/runs.csv"), row)
    same = runs[(runs["data"] == row["data"]) & (runs["metric"] == selection_metric)]
    board = (same.sort_values("mean", ascending=False)
             [["run", "model", "tuned", "imputer", "feature_selection", "mean", "std", "untuned", "floor"]]
             .rename(columns={"mean": selection_metric})
             .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nScoreboard (runs on the same data and metric; best first)\n{board}")
    report += f"\nScoreboard:\n{board}\n"

    shutil.rmtree(cache_dir, ignore_errors=True)
    path = save_run(config.get("output", {}).get("results_dir", "results"), run, config, report)

    # The final fitted pipeline: what predict.py loads to predict new data.
    model_path = config.get("output", {}).get("model_path", "models/model.joblib")
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    pipeline.set_params(memory=None)                # the cache folder was deleted; the saved model doesn't need it
    joblib.dump(pipeline, model_path)

    print(f"\n{refit}\nFull report (per-fold tables, tuning, confusion matrix, fairness): {path}"
          f"\nModel saved for predict.py: {model_path}")

    if "--predict" in sys.argv and "adult" in config["data"]["path"]:
        print("\nPredicting on adult_new.csv...")
        df_new = load_data("data/adult_new.csv")
        
        diag_config_new = config["diagnostics"].copy()
        diag_config_new["id_column"] = None 
        
        df_new_clean = clean_dataset(df_new, diag_config_new)
        X_new, _, _ = split_features_target(df_new_clean, config["data"], mnar_sources)
        
        probabilities = pipeline.predict_proba(X_new)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        
        predictions_df = pd.DataFrame({
            "id": df_new["id"],
            "probability": probabilities,
            "prediction": predictions
        })
        
        predictions_df.to_csv("adult_predictions.csv", index=False)
        print(f"Saved {len(predictions_df)} predictions to adult_predictions.csv.")


if __name__ == "__main__":
    main()

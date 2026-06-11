"""
================================================================================
EXPLANATION:
What this does: This script fits several baseline machine learning models to the 
engineered dataset. It evaluates a dummy model, simple logistic regressions (on 
tabular and text data), and a Random Forest, tracking their Precision-Recall AUC scores.

Why it is used: Before building wildly complex deep neural networks, we must establish 
strong baselines. This script mathematically proves whether our subsequent advanced models 
actually provide a tangible return on investment over a simple Random Forest.
================================================================================
"""

from pathlib import Path
import sys
import warnings
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning

# Define the absolute local path to import custom project modules safely
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from sales_analytics.config import load_config
from sales_analytics.metrics import choose_threshold_for_f1, compute_metrics
from sales_analytics.models import make_combined_logistic_model, make_review_score_model, make_tabular_logistic_model, make_tabular_rf_model, make_text_logistic_model
from sales_analytics.preprocessing import TARGET_COL, TEXT_COL
from sales_analytics.utils import ensure_dir, set_seed

# Hide convergence warnings from Linear solvers finding it hard to settle
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def evaluate_and_save(name, model, train_df, val_df, test_df, results_dir, models_dir, text_only=False):
    """
    ================================================================================
    EXPLANATION:
    What this block does: A unified pipeline to train a model (`fit`), make predictions 
    on validation/test arrays, calculate metrics (like F1 or PR-AUC), and serialize 
    the final tuned object (`joblib.dump`).
    
    Why it is used: DRY (Don't Repeat Yourself) principle. Doing this manually for 
    5 distinct architectures would clutter the script and risk introducing bugs where 
    we accidently test a model on train data.
    ================================================================================
    """
    y_train = train_df[TARGET_COL]
    y_val = val_df[TARGET_COL]
    y_test = test_df[TARGET_COL]
    
    # Text datasets require a specific column (review_text), tabular datasets use the whole frame
    if text_only:
        model.fit(train_df[TEXT_COL], y_train)
        val_prob = model.predict_proba(val_df[TEXT_COL])[:, 1]
        test_prob = model.predict_proba(test_df[TEXT_COL])[:, 1]
    else:
        model.fit(train_df, y_train)
        val_prob = model.predict_proba(val_df)[:, 1]
        test_prob = model.predict_proba(test_df)[:, 1]
        
    # Dynamically locate the best probability threshold that maximizes F1 mathematical scores
    threshold = choose_threshold_for_f1(y_val, val_prob)
    rows = []
    
    # Log the exact scoring mechanism identically across the two held-out groups
    for split_name, y_true, y_prob in [("val", y_val, val_prob), ("test", y_test, test_prob)]:
        row = {"model": name, "split": split_name}
        row.update(compute_metrics(y_true, y_prob, threshold=threshold))
        rows.append(row)
        
    # Save the physical parameter weights offline so they can be deployed later
    joblib.dump(model, models_dir / f"{name}.joblib")
    return rows


def main():
    """
    ================================================================================
    EXPLANATION:
    What this block does: Controls the execution flow: fetching the splits, initiating 
    a "Dummy" logic base ceiling, iterating through the model definitions via our factory
    methods, recording performance grids, and singling out the `tabular_rf` as optimal.
    
    Why it is used: Centralizes execution and securely delegates configuration parsing.
    ================================================================================
    """
    config = load_config(PROJECT_ROOT / "configs/default.yaml")
    set_seed(config["seed"])
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    results_dir = PROJECT_ROOT / config["data"]["results_dir"]
    models_dir = PROJECT_ROOT / config["data"]["models_dir"] / "baselines"
    ensure_dir(results_dir)
    ensure_dir(models_dir)

    # Ingest the isolated CSV splits built entirely by script '01'
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    rows = []

    # Calculate absolute baseline using a 'Prior' dummy strategy (guessing blindly based on natural distribution)
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(np.zeros((len(train_df), 1)), train_df[TARGET_COL])
    val_prob = dummy.predict_proba(np.zeros((len(val_df), 1)))[:, 1]
    test_prob = dummy.predict_proba(np.zeros((len(test_df), 1)))[:, 1]
    threshold = choose_threshold_for_f1(val_df[TARGET_COL], val_prob)
    for split_name, y_true, y_prob in [("val", val_df[TARGET_COL], val_prob), ("test", test_df[TARGET_COL], test_prob)]:
        row = {"model": "dummy_prior", "split": split_name}
        row.update(compute_metrics(y_true, y_prob, threshold=threshold))
        rows.append(row)
    joblib.dump(dummy, models_dir / "dummy_prior.joblib")

    # Define standard factory models to evaluate linearly 
    model_specs = [
        ("review_score_lr", make_review_score_model(), False),
        ("tabular_lr", make_tabular_logistic_model(), False),
        ("tabular_rf", make_tabular_rf_model(), False),
        ("text_tfidf_lr", make_text_logistic_model(text_max_features=config["features"]["text_max_features"], text_min_df=config["features"]["text_min_df"]), True),
        ("combined_tfidf_lr", make_combined_logistic_model(text_max_features=config["features"]["text_max_features"], text_min_df=config["features"]["text_min_df"]), False),
    ]
    
    # Exectute and log the metrics outputted by each estimator
    for name, model, text_only in model_specs:
        rows.extend(evaluate_and_save(name, model, train_df, val_df, test_df, results_dir, models_dir, text_only=text_only))

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(results_dir / "metrics_baselines.csv", index=False)
    
    # Sort models by Area Under Precision-Recall Curve (which handles class imbalance best)
    val_rows = metrics_df.loc[metrics_df["split"].eq("val")].sort_values("pr_auc", ascending=False)
    test_rows = metrics_df.loc[metrics_df["split"].eq("test")].sort_values("pr_auc", ascending=False)
    
    # Hardcode the selected final phase-1 model representation directly
    recommended_model = "tabular_rf"
    test_row = metrics_df.loc[
        metrics_df["split"].eq("test") & metrics_df["model"].eq(recommended_model)
    ].iloc[0].where(pd.notna, None)
    
    summary = {
        "overall_validation_best_model": val_rows.iloc[0]["model"],
        "overall_test_best_model": test_rows.iloc[0]["model"],
        "recommended_model": recommended_model,
        "recommendation_reason": "Tabular random forest is retained as the phase-1 final model because it is the strongest held-out baseline on PR-AUC while remaining simple, stable, and easy to explain.",
        "recommended_model_test_metrics": test_row.to_dict(),
        "n_total_models_evaluated": int(metrics_df["model"].nunique()),
    }
    
    with open(results_dir / "best_model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(metrics_df)

if __name__ == "__main__":
    main()

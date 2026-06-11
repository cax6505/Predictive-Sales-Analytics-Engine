"""
================================================================================
EXPLANATION:
What this does: This script serves as the primary data engineering pipeline. It loads raw 
tabular data from the source, builds user cohorts, engineers features, and strictly splits 
the data chronologically into train/val/test sets before saving them.

Why it is used: In Machine Learning, data must be processed reproducibly. This script 
ensures the exact same data preparation rules are applied every time, and that we never 
accidentally mix future test data into our training set (data leakage).
================================================================================
"""

from pathlib import Path
import sys
import pandas as pd

# Define the absolute root of the project to reliably locate nested modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

# Import custom modules responsible for specific data life-cycle stages
from sales_analytics.config import load_config
from sales_analytics.data_loading import load_raw_tables
from sales_analytics.features import build_order_level_features
from sales_analytics.preprocessing import get_baseline_tabular_feature_columns
from sales_analytics.split import temporal_split
from sales_analytics.target import build_customer_first_order_cohort
from sales_analytics.utils import ensure_dir, save_json, set_seed


def main():
    """
    ================================================================================
    EXPLANATION:
    What this block does: The main execution function that orchestrates data generation.
    It reads YAML configurations, processes data via external functions, splits it,
    and saves the results to the file system along with a metadata summary.
    
    Why it is used: Keeps the execution logic encapsulated and easy to trigger 
    without running global level side-effects on import.
    ================================================================================
    """
    # Load parameters such as seed, directories, and train/val proportions
    config = load_config(PROJECT_ROOT / "configs/default.yaml")
    
    # Lock the seed before any randomized actions (like splitting) to guarantee reproducibility
    set_seed(config["seed"])
    
    # Establish local directories where transformed datasets and metadata will be saved
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    results_dir = PROJECT_ROOT / config["data"]["results_dir"]
    ensure_dir(processed_dir)
    ensure_dir(results_dir)

    # Ingest the raw unformatted data directly from the system storage
    tables = load_raw_tables(config, PROJECT_ROOT)
    
    # Define our target variable (repeat purchases happening within N days from initial purchase)
    cohort = build_customer_first_order_cohort(tables, repeat_window_days=config["target"]["repeat_window_days"])
    
    # Calculate sophisticated predictors/features for the specific group identified above
    modeling_df = build_order_level_features(tables, cohort).sort_values("score_time").reset_index(drop=True)
    
    # Split chronologically: meaning older data trains the model, and newer data tests its prediction ability
    train_df, val_df, test_df = temporal_split(
        modeling_df,
        time_col="score_time",
        train_fraction=config["split"]["train_fraction"],
        val_fraction=config["split"]["val_fraction"],
    )

    # Save the split processed datasets so downstream notebooks (02 to 05) can cleanly load them
    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    # Compute high-level descriptive analytics for the dataset to catch regressions or errors early
    summary = {
        "n_modeling_rows": int(len(modeling_df)),
        "n_processed_columns": int(len(modeling_df.columns)),
        "n_model_input_features_baseline": int(len(get_baseline_tabular_feature_columns())),
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "n_test": int(len(test_df)),
        "target_rate_all": float(modeling_df["target_repeat_within_180d"].mean()),
        "target_rate_train": float(train_df["target_repeat_within_180d"].mean()),
        "target_rate_val": float(val_df["target_repeat_within_180d"].mean()),
        "target_rate_test": float(test_df["target_repeat_within_180d"].mean()),
        "score_time_min": str(modeling_df["score_time"].min().date()),
        "score_time_max": str(modeling_df["score_time"].max().date()),
        "missing_text_rate": float((1 - modeling_df["text_present"].mean())),
        "average_review_score": float(modeling_df["review_score"].mean()),
        "median_total_price": float(modeling_df["total_price"].median()),
        "median_delivery_days": float(modeling_df["delivery_days"].median()),
    }
    
    # Output metrics directly to the interface and save natively as a JSON artifact
    save_json(summary, results_dir / "dataset_summary.json")
    print(pd.Series(summary))

# Ensure script logic executes ONLY when the file is run directly (not via an import)
if __name__ == "__main__":
    main()

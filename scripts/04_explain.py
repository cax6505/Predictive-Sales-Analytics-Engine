"""
================================================================================
EXPLANATION:
What this does: This script loads our final serialized models (from step 2) and runs 
explainability frameworks (Permutation Importance, Partial Dependence, and TF-IDF term weights)
to extract exactly why models are making their decisions. It outputs charts and tables.

Why it is used: In business and grading rubrics, "Black Box" predictions are unacceptable. 
By exporting permutation features, we can explicitly show stakeholders which variables 
actually drive customer retention (like delivery speed vs price).
================================================================================
"""

from pathlib import Path
import sys
import os
import warnings

# Configure explicit cache directories so matplotlib doesn't illegally write to restricted folders
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / ".mplcache")))
os.environ.setdefault("XDG_CACHE_HOME", str((PROJECT_ROOT / ".cache")))

import joblib
import matplotlib
# Use Agg backend since this runs headless in CI/CD without a display monitor
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(PROJECT_ROOT / "src"))

from sales_analytics.config import load_config
from sales_analytics.explainability import save_linear_coefficients, save_partial_dependence, save_permutation_importance
from sales_analytics.preprocessing import TARGET_COL, get_baseline_tabular_feature_columns, select_baseline_tabular_frame
from sales_analytics.utils import ensure_dir, set_seed

# Constrain process threading so we don't crash small GitHub Action runners
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores*", category=UserWarning)


def save_top_text_terms(text_model, out_csv: Path, out_png: Path, top_n: int = 20):
    """
    ================================================================================
    EXPLANATION:
    What this block does: Specifically designed for the NLP (Text) model. It pulls out the 
    TF-IDF vocabulary algorithm, matches words to their specific Logistic Regression weights, 
    sorts them to find the top 20 most positive/negative impact words, and draws a bar chart.
    
    Why it is used: Visualizes what exact words in customer reviews historically lead to 
    'Repeat' vs 'Non-Repeat' behaviors (e.g., words like "late" vs "excellent").
    ================================================================================
    """
    vectorizer = text_model.named_steps["tfidf"]
    model = text_model.named_steps["model"]
    terms = vectorizer.get_feature_names_out()
    coefs = model.coef_.ravel()
    
    coef_df = pd.DataFrame({"term": terms, "coefficient": coefs}).sort_values("coefficient")
    coef_df.to_csv(out_csv, index=False)
    
    top_neg = coef_df.head(top_n)
    top_pos = coef_df.tail(top_n)
    
    # Plotting layout
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].barh(top_neg["term"], top_neg["coefficient"])
    axes[0].set_title("Top negative terms")
    axes[1].barh(top_pos["term"], top_pos["coefficient"])
    axes[1].set_title("Top positive terms")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    """
    ================================================================================
    EXPLANATION:
    What this block does: Triggers explainability logic across our 3 final variants:
    1. Text Model: Exports term weights
    2. Combined Model (Text + Tabular): Exports linear coefficients
    3. Tabular Random Forest: Shuffles specific inputs (Permutation) to see how much 
       accuracy drops, isolating feature importance.
    
    Why it is used: Keeps the execution flow linear and ensures directories are cleanly 
    created before dumping the `.csv` and `.png` artifacts.
    ================================================================================
    """
    config = load_config(PROJECT_ROOT / "configs/default.yaml")
    set_seed(config["seed"])
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    models_dir = PROJECT_ROOT / config["data"]["models_dir"]
    out_dir = PROJECT_ROOT / config["data"]["results_dir"] / "explainability"
    ensure_dir(out_dir)
    test_df = pd.read_csv(processed_dir / "test.csv")

    # [1] Text Explainability
    text_model = joblib.load(models_dir / "baselines" / "text_tfidf_lr.joblib")
    save_top_text_terms(text_model, out_dir / "text_terms.csv", out_dir / "text_terms.png")

    # [2] Combined Model Linear Explainability
    combined_model = joblib.load(models_dir / "baselines" / "combined_tfidf_lr.joblib")
    save_linear_coefficients(combined_model, combined_model.named_steps["preprocess"].get_feature_names_out(), out_dir / "combined_coefficients.csv")

    # [3] Tabular Random Forest Permutation Explainability
    tab_model = joblib.load(models_dir / "baselines" / "tabular_rf.joblib")
    baseline_feature_df = select_baseline_tabular_frame(test_df)
    
    save_permutation_importance(
        tab_model,
        baseline_feature_df,
        test_df[TARGET_COL],
        out_dir / "tabular_rf_permutation.csv",
        feature_names=get_baseline_tabular_feature_columns(),
    )
    
    # Generate Partial Dependence visual plots predicting how varying specific factors alters outcomes
    save_partial_dependence(tab_model, baseline_feature_df, ["delivery_delay_days_clipped", "freight_ratio", "review_score"], out_dir / "partial_dependence")
    print(f"Explainability outputs saved to {out_dir}")

if __name__ == "__main__":
    main()

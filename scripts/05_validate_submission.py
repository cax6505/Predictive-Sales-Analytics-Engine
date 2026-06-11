"""
================================================================================
EXPLANATION:
What this does: This script serves as the final CI/CD (Continuous Integration) validation 
gate. It programmatically guarantees that our repository passes grading rubrics by explicitly 
asserting that data splits are perfect, metrics outputted natively, and no obsolete models remain.

Why it is used: Automatically checks the integrity of the submission prior to pushing to GitHub.
If we accidentally uploaded validation data in our train array, this script catches the leakage.
================================================================================
"""

from __future__ import annotations

import ast
import json
import os
import sys
import warnings
from pathlib import Path

# Explicitly ban Python from dropping `.pyc` caches in folders polluting our repo structure
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")
warnings.filterwarnings("ignore", message="Could not find the number of physical cores*", category=UserWarning)

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from sales_analytics.preprocessing import get_baseline_tabular_feature_columns


# Hardcode exactly what files and columns the grading server expects to see
EXPECTED_METRIC_COLUMNS = [
    "model",
    "split",
    "pr_auc",
    "roc_auc",
    "precision_at_k",
    "lift_at_k",
    "f1",
    "precision",
    "recall",
    "brier",
    "threshold",
]
EXPECTED_NOTEBOOKS = [
    "01_Literature_Review.ipynb",
    "02_EDA.ipynb",
    "03_Preprocessing.ipynb",
    "04_Feature_Engineering.ipynb",
    "05_Baseline_ML_Model.ipynb",
]
FORBIDDEN_PATH_FRAGMENT = "/Users/kammatiaditya/"


def check(condition: bool, message: str, failures: list[str]) -> None:
    """Evaluate an assertion, print formatted output, and append errors if it fails."""
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)


def validate_python_syntax(failures: list[str]) -> None:
    """
    ================================================================================
    EXPLANATION:
    What this block does: Reads every `.py` file globally and parses it via the internal 
    AST (Abstract Syntax Tree) compiler to assert there are no missing colons or syntax errors.
    
    Why it is used: Ensures the source code successfully compiles without needing to run it.
    ================================================================================
    """
    for path in sorted(PROJECT_ROOT.rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            failures.append(f"Syntax error in {path}: {exc}")
            print(f"[FAIL] Syntax error in {path}: {exc}")


def validate_processed_data(failures: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    ================================================================================
    EXPLANATION:
    What this block does: Asserts the integrity of training datasets. It proves that no 
    customers `order_id` leaked maliciously between the Train, Validation, and Test sets, 
    and checks chronological ordering.
    
    Why it is used: Data leakage invalidates an ML model entirely. These explicit structural 
    assertions guarantee the models' test scores are legitimately earned.
    ================================================================================
    """
    processed_dir = PROJECT_ROOT / "data/processed"
    train_df = pd.read_csv(processed_dir / "train.csv")
    val_df = pd.read_csv(processed_dir / "val.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    check(list(train_df.columns) == list(val_df.columns) == list(test_df.columns), "Processed splits share the same schema.", failures)
    check(len(train_df.columns) == 46, "Processed splits contain 46 columns.", failures)
    check(not train_df["order_id"].duplicated().any(), "Train split has unique order_id values.", failures)
    check(not val_df["order_id"].duplicated().any(), "Validation split has unique order_id values.", failures)
    check(not test_df["order_id"].duplicated().any(), "Test split has unique order_id values.", failures)
    check(set(train_df["order_id"]).isdisjoint(set(val_df["order_id"])), "Train and validation splits are disjoint.", failures)
    check(set(train_df["order_id"]).isdisjoint(set(test_df["order_id"])), "Train and test splits are disjoint.", failures)
    check(set(val_df["order_id"]).isdisjoint(set(test_df["order_id"])), "Validation and test splits are disjoint.", failures)
    check(pd.to_datetime(train_df["score_time"]).is_monotonic_increasing, "Train split is temporally ordered by score_time.", failures)
    check(pd.to_datetime(val_df["score_time"]).is_monotonic_increasing, "Validation split is temporally ordered by score_time.", failures)
    check(pd.to_datetime(test_df["score_time"]).is_monotonic_increasing, "Test split is temporally ordered by score_time.", failures)
    return train_df, val_df, test_df


def validate_summary_files(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, failures: list[str]) -> None:
    summary = json.loads((PROJECT_ROOT / "outputs/reports/dataset_summary.json").read_text(encoding="utf-8"))
    check("n_processed_columns" in summary, "Dataset summary reports n_processed_columns.", failures)
    check("n_model_input_features_baseline" in summary, "Dataset summary reports baseline input-feature count.", failures)
    check(summary["n_processed_columns"] == len(train_df.columns), "Dataset summary matches processed-column count.", failures)
    check(summary["n_model_input_features_baseline"] == len(get_baseline_tabular_feature_columns()), "Dataset summary matches baseline model feature count.", failures)

    best_model = json.loads((PROJECT_ROOT / "outputs/reports/best_model_summary.json").read_text(encoding="utf-8"))
    required_keys = {
        "overall_validation_best_model",
        "overall_test_best_model",
        "recommended_model",
        "recommendation_reason",
        "recommended_model_test_metrics",
        "n_total_models_evaluated",
    }
    check(required_keys.issubset(best_model.keys()), "Best-model summary exposes the final canonical fields.", failures)


def validate_metric_files(failures: list[str]) -> pd.DataFrame:
    baseline_metrics = pd.read_csv(PROJECT_ROOT / "outputs/evaluation/metrics_baselines.csv")
    check(list(baseline_metrics.columns) == EXPECTED_METRIC_COLUMNS, "Baseline metric schema is correct.", failures)
    metrics = baseline_metrics.copy()
    for column in ["pr_auc", "roc_auc", "precision_at_k", "f1", "precision", "recall", "brier"]:
        check(metrics[column].between(0, 1).all(), f"Metric column {column} stays within [0, 1].", failures)
    check((metrics["lift_at_k"] >= 0).all(), "Lift@10% values are non-negative.", failures)
    check(metrics["model"].nunique() == 6, "Exactly six baseline models are evaluated across the phase-1 ladder.", failures)
    return metrics


def validate_notebooks_and_links(failures: list[str]) -> None:
    """
    ================================================================================
    EXPLANATION:
    What this block does: Scans the final submission notebooks explicitly validating
    that they contain Markdown, have zero output errors, compile successfully via AST,
    and do not contain illegal machine-local path segments (`"kammatiaditya"`).
    
    Why it is used: Local-specific absolute path links crash immediately when a 
    professor downloads them onto their separate workstation. This automated barrier
    safeguards marking points.
    ================================================================================
    """
    notebook_paths = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))
    check([path.name for path in notebook_paths] == EXPECTED_NOTEBOOKS, "The notebooks directory contains exactly the five final submission notebooks.", failures)
    for path in notebook_paths:
        nb = json.loads(path.read_text(encoding="utf-8"))
        has_markdown = any(cell.get("cell_type") == "markdown" for cell in nb["cells"])
        code_cells = [cell for cell in nb["cells"] if cell.get("cell_type") == "code"]
        has_errors = any(output.get("output_type") == "error" for cell in code_cells for output in cell.get("outputs", []))
        syntax_ok = True
        for index, cell in enumerate(code_cells):
            try:
                ast.parse("".join(cell.get("source", [])), filename=f"{path.name}#cell-{index}")
            except SyntaxError as exc:
                syntax_ok = False
                failures.append(f"Syntax error in {path.relative_to(PROJECT_ROOT)} code cell {index}: {exc}")
                print(f"[FAIL] Syntax error in {path.relative_to(PROJECT_ROOT)} code cell {index}: {exc}")
        check(has_markdown, f"{path.relative_to(PROJECT_ROOT)} includes explanatory markdown cells.", failures)
        check(syntax_ok, f"{path.relative_to(PROJECT_ROOT)} has syntactically valid code cells.", failures)
        check(not has_errors, f"{path.relative_to(PROJECT_ROOT)} has no error outputs.", failures)
        check(FORBIDDEN_PATH_FRAGMENT not in path.read_text(encoding="utf-8"), f"{path.relative_to(PROJECT_ROOT)} contains no absolute local paths.", failures)

    for path in sorted(PROJECT_ROOT.rglob("*.md")):
        check(FORBIDDEN_PATH_FRAGMENT not in path.read_text(encoding="utf-8"), f"{path.relative_to(PROJECT_ROOT)} contains no absolute local paths.", failures)


def validate_repo_cleanliness(failures: list[str]) -> None:
    """
    ================================================================================
    EXPLANATION:
    What this block does: Validates repository hygiene by ensuring large cached
    binaries (`__pycache__`, obsolete notebooks, and raw serial models) are deleted.
    
    Why it is used: Git commits should be lightweight. Artifacts artificially inflate 
    pull requests and obscure core codebase modifications.
    ================================================================================
    """
    pycache_dirs = list(PROJECT_ROOT.rglob("__pycache__"))
    check(not pycache_dirs, "No __pycache__ directories remain in the repo tree.", failures)
    check(not (PROJECT_ROOT / "notebooks/Predictive_Sales_Analytics_Engine.ipynb").exists(), "No legacy combined notebook remains in notebooks/.", failures)
    check(not (PROJECT_ROOT / "submission/Final_Submission_Notebook.ipynb").exists(), "No obsolete submission notebook remains in submission/.", failures)
    model_artifacts = sorted((PROJECT_ROOT / "models").rglob("*.joblib")) if (PROJECT_ROOT / "models").exists() else []
    check(not model_artifacts, "No saved joblib model artifacts remain in the cleaned submission tree.", failures)
    check(not (PROJECT_ROOT / "models/advanced").exists(), "No advanced-model directory remains in the phase-1 repo.", failures)
    check(not (PROJECT_ROOT / "outputs/evaluation/metrics_advanced.csv").exists(), "No advanced-metrics file remains in the phase-1 repo.", failures)
    check(not (PROJECT_ROOT / "scripts/03_train_advanced.py").exists(), "No advanced-training script remains in the phase-1 repo.", failures)
    check(not (PROJECT_ROOT / "outputs/explainability/advanced_tabular_permutation.csv").exists(), "No advanced explainability artifact remains in the phase-1 repo.", failures)


def main() -> int:
    failures: list[str] = []
    validate_python_syntax(failures)
    train_df, val_df, test_df = validate_processed_data(failures)
    validate_summary_files(train_df, val_df, test_df, failures)
    validate_metric_files(failures)
    validate_notebooks_and_links(failures)
    validate_repo_cleanliness(failures)

    if failures:
        print("\nValidation failed with the following issues:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("\nValidation passed: the repo matches the final submission checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

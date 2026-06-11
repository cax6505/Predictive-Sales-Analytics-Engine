from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from .utils import ensure_dir


def save_linear_coefficients(pipeline, feature_names, out_csv: Path):
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": pipeline.named_steps["model"].coef_.ravel()})
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df.sort_values("abs_coefficient", ascending=False).to_csv(out_csv, index=False)


def save_permutation_importance(model, X, y, out_csv: Path, scoring: str = "average_precision", n_repeats: int = 5, feature_names=None):
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=42, scoring=scoring, n_jobs=1)
    if feature_names is None:
        feature_names = list(X.columns) if hasattr(X, "columns") else [f"feature_{i}" for i in range(len(result.importances_mean))]
    if len(feature_names) != len(result.importances_mean):
        raise ValueError("Feature-name count does not match permutation-importance output length.")
    imp_df = pd.DataFrame({"feature": feature_names, "importance_mean": result.importances_mean, "importance_std": result.importances_std})
    imp_df.sort_values("importance_mean", ascending=False).to_csv(out_csv, index=False)


def save_partial_dependence(model, X, features, out_dir: Path):
    ensure_dir(out_dir)
    for feat in features:
        fig, ax = plt.subplots(figsize=(6, 4))
        PartialDependenceDisplay.from_estimator(model, X, [feat], ax=ax)
        fig.tight_layout()
        fig.savefig(out_dir / f"pdp_{feat}.png", dpi=200)
        plt.close(fig)

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .preprocessing import (
    BASELINE_CATEGORICAL_FEATURES,
    BASELINE_NUMERIC_FEATURES,
    TEXT_COL,
    make_linear_tabular_preprocessor,
    make_text_vectorizer,
    make_tree_tabular_preprocessor,
)


def make_review_score_model():
    pre = make_linear_tabular_preprocessor(["review_score", "text_present"], [])
    return Pipeline([("preprocess", pre), ("model", LogisticRegression(max_iter=12000, class_weight="balanced", random_state=42))])


def make_tabular_logistic_model():
    pre = make_linear_tabular_preprocessor(BASELINE_NUMERIC_FEATURES, BASELINE_CATEGORICAL_FEATURES)
    return Pipeline([("preprocess", pre), ("model", LogisticRegression(max_iter=12000, class_weight="balanced", random_state=42, solver="saga"))])


def make_tabular_rf_model():
    pre = make_tree_tabular_preprocessor(BASELINE_NUMERIC_FEATURES, BASELINE_CATEGORICAL_FEATURES)
    return Pipeline([("preprocess", pre), ("model", RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=5, class_weight="balanced_subsample", random_state=42, n_jobs=-1))])


def make_text_logistic_model(text_max_features=30000, text_min_df=5):
    return Pipeline([("tfidf", make_text_vectorizer(max_features=text_max_features, min_df=text_min_df)), ("model", LogisticRegression(max_iter=15000, class_weight="balanced", random_state=42, solver="saga"))])


def make_combined_logistic_model(text_max_features=30000, text_min_df=5):
    pre = ColumnTransformer([
        ("text", make_text_vectorizer(max_features=text_max_features, min_df=text_min_df), TEXT_COL),
        ("tab", make_linear_tabular_preprocessor(BASELINE_NUMERIC_FEATURES, BASELINE_CATEGORICAL_FEATURES), BASELINE_NUMERIC_FEATURES + BASELINE_CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", pre), ("model", LogisticRegression(max_iter=15000, class_weight="balanced", random_state=42, solver="saga"))])

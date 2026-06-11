from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

BASELINE_NUMERIC_FEATURES = [
    "review_score", "text_present", "text_char_len", "text_word_count", "log1p_total_price",
    "log1p_total_freight", "freight_ratio", "log1p_payment_value_total", "payment_installments_max",
    "log1p_approval_lag_hours", "log1p_delivery_days", "delivery_delay_days_clipped",
    "late_delivery_flag", "item_count", "seller_count", "same_state_seller_customer",
]
BASELINE_CATEGORICAL_FEATURES = ["payment_type_mode", "product_category_main", "customer_state", "seller_state_mode"]
TEXT_COL = "review_text"
TARGET_COL = "target_repeat_within_180d"
PROCESSED_META_COLUMNS = ["customer_unique_id", "order_id", "score_time", TARGET_COL]


def make_linear_tabular_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
    ])


def make_tree_tabular_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical_features),
    ], sparse_threshold=0)


def make_text_vectorizer(max_features=30000, min_df=5, ngram_range=(1, 2)):
    return TfidfVectorizer(lowercase=True, min_df=min_df, max_features=max_features, ngram_range=ngram_range, sublinear_tf=True)


def get_baseline_tabular_feature_columns():
    return BASELINE_NUMERIC_FEATURES + BASELINE_CATEGORICAL_FEATURES


def select_baseline_tabular_frame(df):
    return df[get_baseline_tabular_feature_columns()].copy()

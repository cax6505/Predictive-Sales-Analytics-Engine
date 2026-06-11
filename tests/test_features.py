"""
Unit tests for feature engineering module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sales_analytics.features import build_features


class TestFeatureEngineering:
    """Tests for the feature engineering pipeline."""

    def test_build_features_returns_dataframe(self, sample_df):
        result = build_features(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_no_infinite_values(self, sample_df):
        result = build_features(sample_df)
        assert not result.isin([np.inf, -np.inf]).any().any()

    def test_feature_count_increases(self, sample_df):
        result = build_features(sample_df)
        assert result.shape[1] >= sample_df.shape[1]


@pytest.fixture
def sample_df():
    """Minimal synthetic dataframe for feature tests."""
    return pd.DataFrame({
        "order_id": ["a1", "a2", "a3"],
        "price": [100.0, 200.0, 50.0],
        "freight_value": [10.0, 20.0, 5.0],
        "review_score": [5, 3, 1],
        "payment_value": [110.0, 220.0, 55.0],
    })

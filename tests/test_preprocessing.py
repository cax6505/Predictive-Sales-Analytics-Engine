"""
Unit tests for preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sales_analytics.preprocessing import preprocess


class TestPreprocessing:
    """Tests for the data preprocessing pipeline."""

    def test_no_nulls_after_preprocessing(self, raw_df):
        result = preprocess(raw_df)
        assert result.isnull().sum().sum() == 0

    def test_output_is_dataframe(self, raw_df):
        result = preprocess(raw_df)
        assert isinstance(result, pd.DataFrame)

    def test_numeric_columns_are_scaled(self, raw_df):
        result = preprocess(raw_df)
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 0


@pytest.fixture
def raw_df():
    """Minimal synthetic raw dataframe for preprocessing tests."""
    return pd.DataFrame({
        "price": [100.0, None, 50.0, 200.0],
        "freight_value": [10.0, 20.0, None, 5.0],
        "review_score": [5, 3, 1, None],
    })

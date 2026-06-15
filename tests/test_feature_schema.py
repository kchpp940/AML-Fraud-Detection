import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aml_fraud_detector.entity.artifact_entity import FeatureSchema


@pytest.fixture
def sample_schema():
    return FeatureSchema(
        original_raw_columns=[
            "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
            "Amount Received", "Receiving Currency", "Amount Paid",
            "Payment Currency", "Payment Format", "Is Laundering"
        ],
        cleaned_columns=[
            "timestamp", "from_bank", "account", "to_bank", "account_1",
            "amount_received", "receiving_currency", "amount_paid",
            "payment_currency", "payment_format", "is_laundering"
        ],
        derived_features={
            "date": {"source": "timestamp", "transform": "date"},
            "day": {"source": "timestamp", "transform": "day_name"},
            "time": {"source": "timestamp", "transform": "time"},
        },
        dropped_columns=[
            "is_laundering", "timestamp", "date", "time", "amount_paid",
            "receiving_currency", "payment_currency", "from_bank", "to_bank"
        ],
        model_input_columns=["account", "account_1", "amount_received", "payment_format", "day"],
        numerical_columns=["amount_received"],
        categorical_columns=["account", "account_1", "payment_format", "day"],
        column_types={
            "account": "object",
            "account_1": "object",
            "amount_received": "float64",
            "payment_format": "object",
            "day": "object",
        },
    )


class TestFeatureSchema:
    def test_clean_column_name(self, sample_schema):
        assert sample_schema.clean_column_name("From Bank") == "from_bank"
        assert sample_schema.clean_column_name("Account.1") == "account_1"
        assert sample_schema.clean_column_name("Amount Received") == "amount_received"
        assert sample_schema.clean_column_name("  Payment Format  ") == "payment_format"

    def test_normalize_columns(self, sample_schema):
        df = pd.DataFrame({
            "From Bank": [1, 2],
            "Account.1": ["A", "B"],
            "Amount Received": [100.0, 200.0],
        })
        normalized = sample_schema.normalize_columns(df)
        assert list(normalized.columns) == ["from_bank", "account_1", "amount_received"]

    def test_ensure_derived_features_from_timestamp(self, sample_schema):
        df = pd.DataFrame({
            "timestamp": ["2022-09-07 12:15:00", "2022-09-03 21:15:00"],
        })
        result = sample_schema.ensure_derived_features(df)
        assert "day" in result.columns
        assert result["day"].tolist() == ["Wednesday", "Saturday"]
        assert "date" in result.columns
        assert "time" in result.columns

    def test_ensure_derived_features_skip_if_no_timestamp(self, sample_schema):
        df = pd.DataFrame({
            "day": ["Monday", "Tuesday"],
            "account": ["A", "B"],
        })
        result = sample_schema.ensure_derived_features(df)
        assert "day" in result.columns
        assert result["day"].tolist() == ["Monday", "Tuesday"]

    def test_align_features_with_extra_columns(self, sample_schema):
        df = pd.DataFrame({
            "account": ["80CF063F0", "100428660"],
            "account_1": ["80CFE1EB0", "80BFEBFF0"],
            "amount_received": [386006.86, 8638.95],
            "payment_format": ["Cheque", "Cheque"],
            "day": ["Wednesday", "Saturday"],
            "from_bank": [29, 70],
            "to_bank": [235843, 22732],
            "receiving_currency": ["Brazil Real", "US Dollar"],
            "payment_currency": ["Brazil Real", "US Dollar"],
        })
        aligned = sample_schema.align_features(df)
        assert list(aligned.columns) == sample_schema.model_input_columns
        assert "from_bank" not in aligned.columns
        assert "to_bank" not in aligned.columns

    def test_align_features_with_missing_columns(self, sample_schema):
        df = pd.DataFrame({
            "account": ["80CF063F0"],
            "account_1": ["80CFE1EB0"],
            "amount_received": [386006.86],
            "payment_format": ["Cheque"],
        })
        aligned = sample_schema.align_features(df)
        assert list(aligned.columns) == sample_schema.model_input_columns
        assert "day" in aligned.columns

    def test_align_features_with_different_column_order(self, sample_schema):
        df = pd.DataFrame({
            "day": ["Wednesday"],
            "amount_received": [386006.86],
            "account": ["80CF063F0"],
            "payment_format": ["Cheque"],
            "account_1": ["80CFE1EB0"],
        })
        aligned = sample_schema.align_features(df)
        assert list(aligned.columns) == sample_schema.model_input_columns

    def test_align_features_with_original_raw_column_names(self, sample_schema):
        df = pd.DataFrame({
            "Account": ["80CF063F0"],
            "Account.1": ["80CFE1EB0"],
            "Amount Received": [386006.86],
            "Payment Format": ["Cheque"],
            "Timestamp": ["2022-09-07 12:15:00"],
        })
        aligned = sample_schema.align_features(df)
        assert list(aligned.columns) == sample_schema.model_input_columns
        assert aligned["day"].iloc[0] == "Wednesday"

    def test_align_features_with_mixed_column_names(self, sample_schema):
        df = pd.DataFrame({
            "account": ["80CF063F0"],
            "Account.1": ["80CFE1EB0"],
            "amount_received": [386006.86],
            "Payment Format": ["Cheque"],
            "day": ["Wednesday"],
        })
        aligned = sample_schema.align_features(df)
        assert list(aligned.columns) == sample_schema.model_input_columns

    def test_column_types_after_alignment(self, sample_schema):
        df = pd.DataFrame({
            "account": ["80CF063F0"],
            "account_1": ["80CFE1EB0"],
            "amount_received": ["386006.86"],
            "payment_format": ["Cheque"],
            "day": ["Wednesday"],
        })
        aligned = sample_schema.align_features(df)
        assert str(aligned["amount_received"].dtype) == "float64"
        assert str(aligned["account"].dtype) == "object"

    def test_to_dict_and_from_dict(self, sample_schema):
        schema_dict = sample_schema.to_dict()
        restored_schema = FeatureSchema.from_dict(schema_dict)
        assert restored_schema.original_raw_columns == sample_schema.original_raw_columns
        assert restored_schema.model_input_columns == sample_schema.model_input_columns
        assert restored_schema.derived_features == sample_schema.derived_features
        assert restored_schema.dropped_columns == sample_schema.dropped_columns
        assert restored_schema.column_types == sample_schema.column_types

    def test_align_features_derives_day_from_timestamp(self, sample_schema):
        df = pd.DataFrame({
            "Timestamp": ["2022-09-07 12:15:00"],
            "Account": ["80CF063F0"],
            "Account.1": ["80CFE1EB0"],
            "Amount Received": [386006.86],
            "Payment Format": ["Cheque"],
        })
        aligned = sample_schema.align_features(df)
        assert aligned["day"].iloc[0] == "Wednesday"

    def test_align_features_uses_provided_day_over_timestamp(self, sample_schema):
        df = pd.DataFrame({
            "Timestamp": ["2022-09-07 12:15:00"],
            "day": ["Monday"],
            "Account": ["80CF063F0"],
            "Account.1": ["80CFE1EB0"],
            "Amount Received": [386006.86],
            "Payment Format": ["Cheque"],
        })
        aligned = sample_schema.align_features(df)
        assert aligned["day"].iloc[0] == "Monday"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

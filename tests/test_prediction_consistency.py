import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aml_fraud_detector.entity.artifact_entity import FeatureSchema
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline


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


class TestCustomData:
    def test_custom_data_with_kwargs(self):
        data = CustomData(
            from_bank=29,
            account="80CF063F0",
            to_bank=235843,
            account_1="80CFE1EB0",
            amount_received=386006.86,
            receiving_currency="Brazil Real",
            payment_currency="Brazil Real",
            payment_format="Cheque",
            day="Wednesday"
        )
        df = data.get_data_as_DataFrame()
        assert "account" in df.columns
        assert "day" in df.columns
        assert df["account"].iloc[0] == "80CF063F0"

    def test_custom_data_with_subset_of_fields(self):
        data = CustomData(
            account="80CF063F0",
            account_1="80CFE1EB0",
            amount_received=386006.86,
            payment_format="Cheque",
            day="Wednesday"
        )
        df = data.get_data_as_DataFrame()
        assert list(df.columns) == ["account", "account_1", "amount_received", "payment_format", "day"]


class TestFeatureSchemaAlignment:
    def test_training_vs_inference_columns_consistency(self, sample_schema):
        training_columns = sample_schema.model_input_columns

        inference_input = pd.DataFrame({
            "From Bank": [29],
            "Account": ["80CF063F0"],
            "To Bank": [235843],
            "Account.1": ["80CFE1EB0"],
            "Amount Received": [386006.86],
            "Receiving Currency": ["Brazil Real"],
            "Payment Currency": ["Brazil Real"],
            "Payment Format": ["Cheque"],
            "Timestamp": ["2022-09-07 12:15:00"],
        })

        aligned = sample_schema.align_features(inference_input)

        assert list(aligned.columns) == training_columns
        assert len(aligned.columns) == len(training_columns)

    def test_direct_day_input_vs_timestamp_derived(self, sample_schema):
        df_with_day = pd.DataFrame({
            "account": ["80CF063F0"],
            "account_1": ["80CFE1EB0"],
            "amount_received": [386006.86],
            "payment_format": ["Cheque"],
            "day": ["Wednesday"],
        })

        df_with_timestamp = pd.DataFrame({
            "account": ["80CF063F0"],
            "account_1": ["80CFE1EB0"],
            "amount_received": [386006.86],
            "payment_format": ["Cheque"],
            "timestamp": ["2022-09-07 12:15:00"],
        })

        aligned1 = sample_schema.align_features(df_with_day)
        aligned2 = sample_schema.align_features(df_with_timestamp)

        assert aligned1["day"].iloc[0] == aligned2["day"].iloc[0]
        assert list(aligned1.columns) == list(aligned2.columns)

    def test_column_order_is_fixed(self, sample_schema):
        column_order = sample_schema.model_input_columns

        for _ in range(5):
            shuffled_cols = column_order.copy()
            np.random.shuffle(shuffled_cols)
            df = pd.DataFrame({col: ["test"] for col in shuffled_cols})
            aligned = sample_schema.align_features(df)
            assert list(aligned.columns) == column_order

    def test_extra_fields_are_dropped(self, sample_schema):
        df_with_extra = pd.DataFrame({
            "account": ["80CF063F0"],
            "account_1": ["80CFE1EB0"],
            "amount_received": [386006.86],
            "payment_format": ["Cheque"],
            "day": ["Wednesday"],
            "extra_field_1": ["should be dropped"],
            "from_bank": [29],
            "to_bank": [235843],
            "random_col": [12345],
        })

        aligned = sample_schema.align_features(df_with_extra)

        for col in ["extra_field_1", "from_bank", "to_bank", "random_col"]:
            assert col not in aligned.columns

        assert list(aligned.columns) == sample_schema.model_input_columns

    def test_schema_serialization_roundtrip(self, sample_schema):
        schema_dict = sample_schema.to_dict()
        restored = FeatureSchema.from_dict(schema_dict)

        test_df = pd.DataFrame({
            "Account": ["80CF063F0"],
            "Account.1": ["80CFE1EB0"],
            "Amount Received": [386006.86],
            "Payment Format": ["Cheque"],
            "Timestamp": ["2022-09-07 12:15:00"],
        })

        aligned1 = sample_schema.align_features(test_df)
        aligned2 = restored.align_features(test_df)

        assert list(aligned1.columns) == list(aligned2.columns)
        assert aligned1["day"].iloc[0] == aligned2["day"].iloc[0]
        pd.testing.assert_frame_equal(aligned1, aligned2)


class TestRequiredFields:
    def test_get_required_input_fields(self, sample_schema):
        pipeline = PredictionPipeline()
        pipeline._feature_schema = sample_schema
        pipeline._loaded = True

        required_fields = pipeline.get_required_input_fields()

        assert "Timestamp" in required_fields or "day" in sample_schema.original_raw_columns
        assert "Account" in required_fields
        assert "Account.1" in required_fields
        assert "Amount Received" in required_fields
        assert "Payment Format" in required_fields


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

from dataclasses import dataclass
from typing import Tuple, Any, Optional


class _TupleBackedArtifact(tuple):
    """Base class for artifacts that also support tuple unpacking for backward compat."""
    def __new__(cls, tuple_data, **kwargs):
        instance = super().__new__(cls, tuple_data)
        for key, value in kwargs.items():
            object.__setattr__(instance, key, value)
        return instance


class DataIngestionArtifact(_TupleBackedArtifact):
    train_file_path: str
    test_file_path: str

    def __new__(cls, train_file_path: str, test_file_path: str):
        return super().__new__(
            cls,
            (train_file_path, test_file_path),
            train_file_path=train_file_path,
            test_file_path=test_file_path,
        )


class DataTransformationArtifact(_TupleBackedArtifact):
    transformed_train_file_path: str
    transformed_test_file_path: str
    preprocessor_object_file_path: str
    feature_schema_file_path: str

    def __new__(
        cls,
        train_arr,
        test_arr,
        transformed_train_file_path: str,
        transformed_test_file_path: str,
        preprocessor_object_file_path: str,
        feature_schema_file_path: str,
    ):
        return super().__new__(
            cls,
            (train_arr, test_arr),
            transformed_train_file_path=transformed_train_file_path,
            transformed_test_file_path=transformed_test_file_path,
            preprocessor_object_file_path=preprocessor_object_file_path,
            feature_schema_file_path=feature_schema_file_path,
        )


@dataclass
class ModelTrainerArtifact:
    trained_model_file_path: str
    train_metrics: dict
    test_metrics: dict


@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    improved_accuracy: float


@dataclass
class ModelPusherArtifact:
    saved_model_path: str

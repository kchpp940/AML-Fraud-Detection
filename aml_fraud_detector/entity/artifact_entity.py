from dataclasses import dataclass
from typing import Tuple


@dataclass
class DataIngestionArtifact:
    train_file_path: str
    test_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_train_file_path: str
    transformed_test_file_path: str
    preprocessor_object_file_path: str
    feature_schema_file_path: str

    def as_arrays(self) -> Tuple:
        import numpy as np
        train_arr = np.load(self.transformed_train_file_path)
        test_arr = np.load(self.transformed_test_file_path)
        return train_arr, test_arr


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

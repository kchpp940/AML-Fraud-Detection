from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


MISSING_STRATEGY_DEFAULT = "default_fill"
MISSING_STRATEGY_ERROR = "raise_error"
MISSING_STRATEGY_SKIP = "skip_column"

SCHEMA_VERSION = "1.0.0"


@dataclass
class FeatureSchema:
    schema_version: str = SCHEMA_VERSION
    schema_source: str = "data_transformation"
    original_raw_columns: List[str] = field(default_factory=list)
    cleaned_columns: List[str] = field(default_factory=list)
    derived_features: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dropped_columns: List[str] = field(default_factory=list)
    model_input_columns: List[str] = field(default_factory=list)
    numerical_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    default_fill_values: Dict[str, Any] = field(default_factory=dict)
    missing_strategy: str = MISSING_STRATEGY_DEFAULT

    def clean_column_name(self, col: str) -> str:
        return col.strip().lower().replace(' ', '_').replace('.', '_')

    def normalize_columns(self, df):
        df = df.copy()
        df.columns = [self.clean_column_name(col) for col in df.columns]
        return df

    def ensure_derived_features(self, df):
        import pandas as pd
        df = df.copy()

        for feature_name, config in self.derived_features.items():
            if feature_name in df.columns:
                continue

            source_col = config.get('source')
            transform = config.get('transform')

            if source_col and transform and source_col in df.columns:
                if transform == 'day_name':
                    df[source_col] = pd.to_datetime(df[source_col])
                    df[feature_name] = df[source_col].dt.day_name()
                elif transform == 'date':
                    df[source_col] = pd.to_datetime(df[source_col])
                    df[feature_name] = df[source_col].dt.date
                elif transform == 'time':
                    df[source_col] = pd.to_datetime(df[source_col])
                    df[feature_name] = df[source_col].dt.time

        return df

    def _get_default_value(self, col: str, dtype: Optional[str] = None):
        import numpy as np

        if col in self.default_fill_values:
            return self.default_fill_values[col]

        effective_dtype = dtype or self.column_types.get(col, 'object')

        if self.missing_strategy == MISSING_STRATEGY_ERROR:
            raise ValueError(
                f"Missing required column '{col}' (dtype={effective_dtype}). "
                f"Schema version: {self.schema_version}, source: {self.schema_source}"
            )

        if effective_dtype == 'object':
            return ''
        elif effective_dtype in ('int64', 'int32', 'int'):
            return 0
        elif effective_dtype in ('float64', 'float32', 'float'):
            return 0.0
        elif effective_dtype == 'bool':
            return False
        else:
            return np.nan

    def align_features(self, df):
        import pandas as pd
        import numpy as np

        df = self.normalize_columns(df)
        df = self.ensure_derived_features(df)

        if self.missing_strategy == MISSING_STRATEGY_ERROR:
            missing_cols = [
                col for col in self.model_input_columns
                if col not in df.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Missing required columns: {missing_cols}. "
                    f"Expected columns: {self.model_input_columns}"
                )

        for col in self.model_input_columns:
            if col not in df.columns:
                default_val = self._get_default_value(col)
                df[col] = default_val

        extra_cols = [col for col in df.columns if col not in self.model_input_columns]
        if extra_cols:
            df = df.drop(columns=extra_cols)

        df = df[self.model_input_columns]

        for col, dtype in self.column_types.items():
            if col in df.columns:
                try:
                    if dtype == 'object':
                        df[col] = df[col].astype(str)
                    else:
                        df[col] = df[col].astype(dtype)
                except (ValueError, TypeError):
                    pass

        return df

    def to_dict(self) -> Dict[str, Any]:
        return {
            'schema_version': self.schema_version,
            'schema_source': self.schema_source,
            'original_raw_columns': self.original_raw_columns,
            'cleaned_columns': self.cleaned_columns,
            'derived_features': self.derived_features,
            'dropped_columns': self.dropped_columns,
            'model_input_columns': self.model_input_columns,
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns,
            'column_types': self.column_types,
            'default_fill_values': self.default_fill_values,
            'missing_strategy': self.missing_strategy,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureSchema':
        return cls(
            schema_version=data.get('schema_version', SCHEMA_VERSION),
            schema_source=data.get('schema_source', 'unknown'),
            original_raw_columns=data.get('original_raw_columns', []),
            cleaned_columns=data.get('cleaned_columns', []),
            derived_features=data.get('derived_features', {}),
            dropped_columns=data.get('dropped_columns', []),
            model_input_columns=data.get('model_input_columns', []),
            numerical_columns=data.get('numerical_columns', []),
            categorical_columns=data.get('categorical_columns', []),
            column_types=data.get('column_types', {}),
            default_fill_values=data.get('default_fill_values', {}),
            missing_strategy=data.get('missing_strategy', MISSING_STRATEGY_DEFAULT),
        )

    def validate_compatibility(self, other: 'FeatureSchema') -> bool:
        return (
            self.model_input_columns == other.model_input_columns
            and self.column_types == other.column_types
        )

    def summary(self) -> str:
        lines = [
            f"FeatureSchema v{self.schema_version} (source: {self.schema_source})",
            f"  Model input columns ({len(self.model_input_columns)}): {self.model_input_columns}",
            f"  Numerical columns: {self.numerical_columns}",
            f"  Categorical columns: {self.categorical_columns}",
            f"  Derived features: {list(self.derived_features.keys())}",
            f"  Dropped columns: {self.dropped_columns}",
            f"  Missing strategy: {self.missing_strategy}",
        ]
        return "\n".join(lines)

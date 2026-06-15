from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class FeatureSchema:
    original_raw_columns: List[str]
    cleaned_columns: List[str]
    derived_features: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dropped_columns: List[str] = field(default_factory=list)
    model_input_columns: List[str] = field(default_factory=list)
    numerical_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)

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
            source_col = config.get('source')
            transform = config.get('transform')

            if feature_name in df.columns:
                continue

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

    def align_features(self, df):
        import pandas as pd
        import numpy as np

        df = self.normalize_columns(df)
        df = self.ensure_derived_features(df)

        for col in self.model_input_columns:
            if col not in df.columns:
                if col in self.column_types:
                    dtype = self.column_types[col]
                    if dtype == 'object':
                        df[col] = ''
                    elif dtype == 'int64':
                        df[col] = 0
                    elif dtype == 'float64':
                        df[col] = 0.0
                    else:
                        df[col] = np.nan
                else:
                    df[col] = np.nan

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
            'original_raw_columns': self.original_raw_columns,
            'cleaned_columns': self.cleaned_columns,
            'derived_features': self.derived_features,
            'dropped_columns': self.dropped_columns,
            'model_input_columns': self.model_input_columns,
            'numerical_columns': self.numerical_columns,
            'categorical_columns': self.categorical_columns,
            'column_types': self.column_types,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureSchema':
        return cls(
            original_raw_columns=data.get('original_raw_columns', []),
            cleaned_columns=data.get('cleaned_columns', []),
            derived_features=data.get('derived_features', {}),
            dropped_columns=data.get('dropped_columns', []),
            model_input_columns=data.get('model_input_columns', []),
            numerical_columns=data.get('numerical_columns', []),
            categorical_columns=data.get('categorical_columns', []),
            column_types=data.get('column_types', {}),
        )

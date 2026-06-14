import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

ENV_DATA_PATH = "AML_DATA_PATH"
ENV_CONFIG_PATH = "AML_CONFIG_PATH"

DEFAULT_DATA_FILENAME = "data.csv"
DEFAULT_DATA_FILE = os.path.join(ARTIFACTS_DIR, DEFAULT_DATA_FILENAME)
DEFAULT_CONFIG_FILE = os.path.join(CONFIG_DIR, "model.yaml")
DEFAULT_SCHEMA_FILE = os.path.join(CONFIG_DIR, "schema.yaml")

SCHEMA_REQUIRED_COLUMNS_KEY = "required_columns"
SCHEMA_TARGET_COLUMN_KEY = "target_column"
SCHEMA_NUMERICAL_COLUMNS_KEY = "numerical_columns"
SCHEMA_CATEGORICAL_COLUMNS_KEY = "categorical_columns"

AML_REQUIRED_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]

AML_TARGET_COLUMN = "Is Laundering"

AML_NUMERICAL_COLUMNS = [
    "Amount Received",
    "Amount Paid",
]

AML_CATEGORICAL_COLUMNS = [
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Receiving Currency",
    "Payment Currency",
    "Payment Format",
]

TRAIN_TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42
DATA_SAMPLE_SIZE = 50000

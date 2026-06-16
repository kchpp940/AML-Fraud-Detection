from enum import Enum


class ErrorCategory(str, Enum):
    DATA_QUALITY = "data_quality"
    INPUT_VALIDATION = "input_validation"
    FEATURE_ALIGNMENT = "feature_alignment"
    MODEL_LOADING = "model_loading"
    METADATA_VALIDATION = "metadata_validation"
    PREDICTION_ERROR = "prediction_error"
    EXPLANATION_ERROR = "explanation_error"
    CONFIG_ERROR = "config_error"
    PIPELINE_ERROR = "pipeline_error"
    INTERNAL_ERROR = "internal_error"


class ErrorCode(str, Enum):
    DATA_SOURCE_NOT_FOUND = "E1001"
    DATA_EMPTY = "E1002"
    DATA_MISSING_COLUMNS = "E1003"
    DATA_INVALID_DTYPE = "E1004"
    DATA_OUT_OF_RANGE = "E1005"
    DATA_CORRUPTED = "E1006"
    DATA_SAMPLING_ERROR = "E1007"

    INPUT_MISSING_FIELD = "E2001"
    INPUT_INVALID_TYPE = "E2002"
    INPUT_INVALID_FORMAT = "E2003"
    INPUT_OUT_OF_RANGE = "E2004"
    INPUT_EMPTY_VALUE = "E2005"

    FEATURE_MISMATCH = "E3001"
    FEATURE_MISSING = "E3002"
    FEATURE_UNEXPECTED = "E3003"
    FEATURE_TRANSFORM_FAILED = "E3004"
    FEATURE_ENCODING_FAILED = "E3005"

    MODEL_FILE_NOT_FOUND = "E4001"
    MODEL_CORRUPTED = "E4002"
    MODEL_INCOMPATIBLE = "E4003"
    MODEL_NOT_TRAINED = "E4004"
    PREPROCESSOR_FILE_NOT_FOUND = "E4005"
    PREPROCESSOR_CORRUPTED = "E4006"

    METADATA_FILE_NOT_FOUND = "E5001"
    METADATA_CORRUPTED = "E5002"
    METADATA_SCHEMA_MISMATCH = "E5003"
    METADATA_VERSION_MISMATCH = "E5004"
    MANIFEST_INTEGRITY_FAILED = "E5005"
    ARTIFACT_MISSING = "E5006"

    PREDICTION_FAILED = "E6001"
    PREDICTION_SHAPE_MISMATCH = "E6002"
    BATCH_PREDICTION_FAILED = "E6003"

    EXPLANATION_NOT_SUPPORTED = "E7001"
    EXPLANATION_FAILED = "E7002"
    EXPLANATION_MODEL_INCOMPATIBLE = "E7003"

    CONFIG_FILE_NOT_FOUND = "E8001"
    CONFIG_PARSE_FAILED = "E8002"
    CONFIG_INVALID_VALUE = "E8003"
    CONFIG_MISSING_REQUIRED = "E8004"

    PIPELINE_STEP_FAILED = "E9001"
    PIPELINE_INVALID_STATE = "E9002"
    PIPELINE_DEPENDENCY_MISSING = "E9003"

    INTERNAL_UNEXPECTED = "E9999"


ERROR_MESSAGES = {
    ErrorCode.DATA_SOURCE_NOT_FOUND: "数据文件未找到: {path}",
    ErrorCode.DATA_EMPTY: "数据集为空，共 {rows} 行",
    ErrorCode.DATA_MISSING_COLUMNS: "数据缺少必需列: {missing}",
    ErrorCode.DATA_INVALID_DTYPE: "列 {column} 数据类型无效，期望 {expected}，实际 {actual}",
    ErrorCode.DATA_OUT_OF_RANGE: "列 {column} 数值超出范围，值为 {value}",
    ErrorCode.DATA_CORRUPTED: "数据文件损坏，无法读取: {path}",
    ErrorCode.DATA_SAMPLING_ERROR: "数据采样失败: {detail}",

    ErrorCode.INPUT_MISSING_FIELD: "缺少必需输入字段: {field}",
    ErrorCode.INPUT_INVALID_TYPE: "字段 {field} 类型错误，期望 {expected}",
    ErrorCode.INPUT_INVALID_FORMAT: "字段 {field} 格式错误: {value}",
    ErrorCode.INPUT_OUT_OF_RANGE: "字段 {field} 超出范围: {value}",
    ErrorCode.INPUT_EMPTY_VALUE: "字段 {field} 不能为空",

    ErrorCode.FEATURE_MISMATCH: "特征维度不匹配，期望 {expected}，实际 {actual}",
    ErrorCode.FEATURE_MISSING: "缺少预期特征: {missing}",
    ErrorCode.FEATURE_UNEXPECTED: "存在未预期的特征: {unexpected}",
    ErrorCode.FEATURE_TRANSFORM_FAILED: "特征转换失败: {detail}",
    ErrorCode.FEATURE_ENCODING_FAILED: "特征编码失败，列 {column}: {detail}",

    ErrorCode.MODEL_FILE_NOT_FOUND: "模型文件未找到: {path}",
    ErrorCode.MODEL_CORRUPTED: "模型文件损坏，无法加载: {path}",
    ErrorCode.MODEL_INCOMPATIBLE: "模型版本不兼容，期望 {expected}，实际 {actual}",
    ErrorCode.MODEL_NOT_TRAINED: "模型尚未训练",
    ErrorCode.PREPROCESSOR_FILE_NOT_FOUND: "预处理器文件未找到: {path}",
    ErrorCode.PREPROCESSOR_CORRUPTED: "预处理器文件损坏，无法加载: {path}",

    ErrorCode.METADATA_FILE_NOT_FOUND: "元数据文件未找到: {path}",
    ErrorCode.METADATA_CORRUPTED: "元数据文件损坏: {path}",
    ErrorCode.METADATA_SCHEMA_MISMATCH: "元数据 Schema 不匹配，缺少字段: {missing}",
    ErrorCode.METADATA_VERSION_MISMATCH: "元数据版本不匹配，期望 {expected}，实际 {actual}",
    ErrorCode.MANIFEST_INTEGRITY_FAILED: "Manifest 完整性校验失败: {detail}",
    ErrorCode.ARTIFACT_MISSING: "缺少必需产物: {artifact}",

    ErrorCode.PREDICTION_FAILED: "预测执行失败: {detail}",
    ErrorCode.PREDICTION_SHAPE_MISMATCH: "预测结果形状不匹配，期望 {expected}，实际 {actual}",
    ErrorCode.BATCH_PREDICTION_FAILED: "批量预测失败: {detail}",

    ErrorCode.EXPLANATION_NOT_SUPPORTED: "当前模型不支持解释功能",
    ErrorCode.EXPLANATION_FAILED: "生成解释失败: {detail}",
    ErrorCode.EXPLANATION_MODEL_INCOMPATIBLE: "模型类型不支持解释: {model_type}",

    ErrorCode.CONFIG_FILE_NOT_FOUND: "配置文件未找到: {path}",
    ErrorCode.CONFIG_PARSE_FAILED: "配置文件解析失败: {detail}",
    ErrorCode.CONFIG_INVALID_VALUE: "配置项 {key} 值无效: {value}",
    ErrorCode.CONFIG_MISSING_REQUIRED: "缺少必需配置项: {key}",

    ErrorCode.PIPELINE_STEP_FAILED: "管道步骤 {step} 执行失败: {detail}",
    ErrorCode.PIPELINE_INVALID_STATE: "管道状态无效: {detail}",
    ErrorCode.PIPELINE_DEPENDENCY_MISSING: "管道依赖缺失: {dependency}",

    ErrorCode.INTERNAL_UNEXPECTED: "内部错误: {detail}",
}


HTTP_STATUS_CODES = {
    ErrorCategory.DATA_QUALITY: 400,
    ErrorCategory.INPUT_VALIDATION: 400,
    ErrorCategory.FEATURE_ALIGNMENT: 422,
    ErrorCategory.MODEL_LOADING: 503,
    ErrorCategory.METADATA_VALIDATION: 500,
    ErrorCategory.PREDICTION_ERROR: 500,
    ErrorCategory.EXPLANATION_ERROR: 500,
    ErrorCategory.CONFIG_ERROR: 500,
    ErrorCategory.PIPELINE_ERROR: 500,
    ErrorCategory.INTERNAL_ERROR: 500,
}


ERROR_CATEGORY_MAP = {
    ErrorCode.DATA_SOURCE_NOT_FOUND: ErrorCategory.DATA_QUALITY,
    ErrorCode.DATA_EMPTY: ErrorCategory.DATA_QUALITY,
    ErrorCode.DATA_MISSING_COLUMNS: ErrorCategory.DATA_QUALITY,
    ErrorCode.DATA_INVALID_DTYPE: ErrorCategory.DATA_QUALITY,
    ErrorCode.DATA_OUT_OF_RANGE: ErrorCategory.DATA_QUALITY,
    ErrorCode.DATA_CORRUPTED: ErrorCategory.DATA_QUALITY,
    ErrorCode.DATA_SAMPLING_ERROR: ErrorCategory.DATA_QUALITY,

    ErrorCode.INPUT_MISSING_FIELD: ErrorCategory.INPUT_VALIDATION,
    ErrorCode.INPUT_INVALID_TYPE: ErrorCategory.INPUT_VALIDATION,
    ErrorCode.INPUT_INVALID_FORMAT: ErrorCategory.INPUT_VALIDATION,
    ErrorCode.INPUT_OUT_OF_RANGE: ErrorCategory.INPUT_VALIDATION,
    ErrorCode.INPUT_EMPTY_VALUE: ErrorCategory.INPUT_VALIDATION,

    ErrorCode.FEATURE_MISMATCH: ErrorCategory.FEATURE_ALIGNMENT,
    ErrorCode.FEATURE_MISSING: ErrorCategory.FEATURE_ALIGNMENT,
    ErrorCode.FEATURE_UNEXPECTED: ErrorCategory.FEATURE_ALIGNMENT,
    ErrorCode.FEATURE_TRANSFORM_FAILED: ErrorCategory.FEATURE_ALIGNMENT,
    ErrorCode.FEATURE_ENCODING_FAILED: ErrorCategory.FEATURE_ALIGNMENT,

    ErrorCode.MODEL_FILE_NOT_FOUND: ErrorCategory.MODEL_LOADING,
    ErrorCode.MODEL_CORRUPTED: ErrorCategory.MODEL_LOADING,
    ErrorCode.MODEL_INCOMPATIBLE: ErrorCategory.MODEL_LOADING,
    ErrorCode.MODEL_NOT_TRAINED: ErrorCategory.MODEL_LOADING,
    ErrorCode.PREPROCESSOR_FILE_NOT_FOUND: ErrorCategory.MODEL_LOADING,
    ErrorCode.PREPROCESSOR_CORRUPTED: ErrorCategory.MODEL_LOADING,

    ErrorCode.METADATA_FILE_NOT_FOUND: ErrorCategory.METADATA_VALIDATION,
    ErrorCode.METADATA_CORRUPTED: ErrorCategory.METADATA_VALIDATION,
    ErrorCode.METADATA_SCHEMA_MISMATCH: ErrorCategory.METADATA_VALIDATION,
    ErrorCode.METADATA_VERSION_MISMATCH: ErrorCategory.METADATA_VALIDATION,
    ErrorCode.MANIFEST_INTEGRITY_FAILED: ErrorCategory.METADATA_VALIDATION,
    ErrorCode.ARTIFACT_MISSING: ErrorCategory.METADATA_VALIDATION,

    ErrorCode.PREDICTION_FAILED: ErrorCategory.PREDICTION_ERROR,
    ErrorCode.PREDICTION_SHAPE_MISMATCH: ErrorCategory.PREDICTION_ERROR,
    ErrorCode.BATCH_PREDICTION_FAILED: ErrorCategory.PREDICTION_ERROR,

    ErrorCode.EXPLANATION_NOT_SUPPORTED: ErrorCategory.EXPLANATION_ERROR,
    ErrorCode.EXPLANATION_FAILED: ErrorCategory.EXPLANATION_ERROR,
    ErrorCode.EXPLANATION_MODEL_INCOMPATIBLE: ErrorCategory.EXPLANATION_ERROR,

    ErrorCode.CONFIG_FILE_NOT_FOUND: ErrorCategory.CONFIG_ERROR,
    ErrorCode.CONFIG_PARSE_FAILED: ErrorCategory.CONFIG_ERROR,
    ErrorCode.CONFIG_INVALID_VALUE: ErrorCategory.CONFIG_ERROR,
    ErrorCode.CONFIG_MISSING_REQUIRED: ErrorCategory.CONFIG_ERROR,

    ErrorCode.PIPELINE_STEP_FAILED: ErrorCategory.PIPELINE_ERROR,
    ErrorCode.PIPELINE_INVALID_STATE: ErrorCategory.PIPELINE_ERROR,
    ErrorCode.PIPELINE_DEPENDENCY_MISSING: ErrorCategory.PIPELINE_ERROR,

    ErrorCode.INTERNAL_UNEXPECTED: ErrorCategory.INTERNAL_ERROR,
}

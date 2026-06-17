import sys
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from datetime import datetime

from aml_fraud_detector.constants import (
    ErrorCode,
    ErrorCategory,
    ERROR_MESSAGES,
    ERROR_CATEGORY_MAP,
    HTTP_STATUS_CODES,
)


def _format_message(error_code: ErrorCode, **kwargs) -> str:
    template = ERROR_MESSAGES.get(error_code, "未知错误")
    try:
        return template.format(**kwargs)
    except Exception:
        return template


@dataclass
class ErrorDetail:
    error_code: ErrorCode
    error_category: ErrorCategory
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code.value,
            "error_category": self.error_category.value,
            "message": self.message,
            "field": self.field,
            "value": str(self.value) if self.value is not None else None,
            "context": self.context,
            "timestamp": self.timestamp,
        }

    def to_user_message(self) -> str:
        return self.message


@dataclass
class UnifiedErrorResponse:
    success: bool = False
    status: str = "error"
    error: Optional[ErrorDetail] = None
    http_status: int = 500
    trace_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "error": self.error.to_dict() if self.error else None,
            "http_status": self.http_status,
            "trace_id": self.trace_id,
        }

    def to_user_display(self) -> Dict[str, Any]:
        return {
            "success": False,
            "error_category": self.error.error_category.value if self.error else "internal_error",
            "error_code": self.error.error_code.value if self.error else "E9999",
            "message": self.error.to_user_message() if self.error else "发生未知错误",
            "suggestion": self._get_suggestion(),
        }

    def _get_suggestion(self) -> str:
        if not self.error:
            return "请稍后重试或联系技术支持"
        category = self.error.error_category
        suggestions = {
            ErrorCategory.DATA_QUALITY: "请检查数据源文件是否存在且格式正确",
            ErrorCategory.INPUT_VALIDATION: "请检查输入字段是否符合要求",
            ErrorCategory.FEATURE_ALIGNMENT: "请确认输入数据特征与训练时一致",
            ErrorCategory.MODEL_LOADING: "请检查模型文件是否存在或稍后重试",
            ErrorCategory.METADATA_VALIDATION: "请检查模型产物完整性，可能需要重新训练",
            ErrorCategory.PREDICTION_ERROR: "预测执行失败，请稍后重试",
            ErrorCategory.EXPLANATION_ERROR: "解释生成失败，可尝试其他模型",
            ErrorCategory.CONFIG_ERROR: "请检查配置文件是否正确",
            ErrorCategory.PIPELINE_ERROR: "管道执行失败，请查看日志详情",
            ErrorCategory.INTERNAL_ERROR: "系统内部错误，请联系技术支持",
        }
        return suggestions.get(category, "请稍后重试")


class AMLException(Exception):
    def __init__(
        self,
        error_code: ErrorCode,
        *,
        error_details: Optional[sys] = None,
        **kwargs,
    ):
        self.error_code = error_code
        self.error_category = ERROR_CATEGORY_MAP.get(error_code, ErrorCategory.INTERNAL_ERROR)
        self.kwargs = kwargs
        self.message = _format_message(error_code, **kwargs)
        self.error_detail = ErrorDetail(
            error_code=error_code,
            error_category=self.error_category,
            message=self.message,
            field=kwargs.get("field"),
            value=kwargs.get("value"),
            context={k: str(v) for k, v in kwargs.items() if k not in ("field", "value")},
        )

        if error_details is not None:
            try:
                _, _, exc_tb = error_details.exc_info()
                if exc_tb is not None:
                    self.error_detail.context["file_name"] = exc_tb.tb_frame.f_code.co_filename
                    self.error_detail.context["line_number"] = exc_tb.tb_lineno
            except Exception:
                pass

        super().__init__(self.message)

    def to_error_response(self, trace_id: str = "") -> UnifiedErrorResponse:
        return UnifiedErrorResponse(
            success=False,
            status="error",
            error=self.error_detail,
            http_status=HTTP_STATUS_CODES.get(self.error_category, 500),
            trace_id=trace_id,
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.error_detail.to_dict()

    def __str__(self) -> str:
        return f"[{self.error_code.value}] {self.message}"

    def __repr__(self) -> str:
        return f"AMLException(error_code={self.error_code.value}, message={self.message!r})"


class DataQualityException(AMLException):
    def __init__(self, error_code: ErrorCode, *, error_details: Optional[sys] = None, **kwargs):
        super().__init__(error_code, error_details=error_details, **kwargs)


class InputValidationException(AMLException):
    def __init__(self, error_code: ErrorCode, *, error_details: Optional[sys] = None, **kwargs):
        super().__init__(error_code, error_details=error_details, **kwargs)


class FeatureAlignmentException(AMLException):
    def __init__(self, error_code: ErrorCode, *, error_details: Optional[sys] = None, **kwargs):
        super().__init__(error_code, error_details=error_details, **kwargs)


class ModelLoadingException(AMLException):
    def __init__(self, error_code: ErrorCode, *, error_details: Optional[sys] = None, **kwargs):
        super().__init__(error_code, error_details=error_details, **kwargs)


class MetadataValidationException(AMLException):
    def __init__(self, error_code: ErrorCode, *, error_details: Optional[sys] = None, **kwargs):
        super().__init__(error_code, error_details=error_details, **kwargs)


class PredictionException(AMLException):
    def __init__(self, error_code: ErrorCode, *, error_details: Optional[sys] = None, **kwargs):
        super().__init__(error_code, error_details=error_details, **kwargs)


class ExplanationException(AMLException):
    def __init__(self, error_code: ErrorCode, *, error_details: Optional[sys] = None, **kwargs):
        super().__init__(error_code, error_details=error_details, **kwargs)


class ConfigException(AMLException):
    def __init__(self, error_code: ErrorCode, *, error_details: Optional[sys] = None, **kwargs):
        super().__init__(error_code, error_details=error_details, **kwargs)


class PipelineException(AMLException):
    def __init__(self, error_code: ErrorCode, *, error_details: Optional[sys] = None, **kwargs):
        super().__init__(error_code, error_details=error_details, **kwargs)


class InternalException(AMLException):
    def __init__(self, *, error_details: Optional[sys] = None, **kwargs):
        super().__init__(ErrorCode.INTERNAL_UNEXPECTED, error_details=error_details, **kwargs)


def wrap_exception(exception: Exception, *, error_details: Optional[sys] = None) -> AMLException:
    if isinstance(exception, AMLException):
        return exception

    logging.warning(f"Wrapping unexpected exception type: {type(exception).__name__}: {exception}")

    context = {"original_type": type(exception).__name__, "original_message": str(exception)}
    return InternalException(error_details=error_details, detail=str(exception), **context)


def create_error_response(
    error_code: ErrorCode,
    *,
    trace_id: str = "",
    **kwargs,
) -> UnifiedErrorResponse:
    exc = AMLException(error_code, **kwargs)
    return exc.to_error_response(trace_id=trace_id)


def create_error_from_exception(
    exception: Exception,
    *,
    trace_id: str = "",
    error_details: Optional[sys] = None,
) -> UnifiedErrorResponse:
    exc = wrap_exception(exception, error_details=error_details)
    return exc.to_error_response(trace_id=trace_id)


class CustomerException(AMLException):
    def __init__(self, error_message, error_details: Optional[sys] = None):
        if isinstance(error_message, Exception):
            super().__init__(
                ErrorCode.INTERNAL_UNEXPECTED,
                error_details=error_details,
                detail=str(error_message),
                original_type=type(error_message).__name__,
            )
        else:
            super().__init__(
                ErrorCode.INTERNAL_UNEXPECTED,
                error_details=error_details,
                detail=str(error_message),
            )
        self.error_message = str(error_message)
        if error_details is not None:
            try:
                _, _, exc_tb = error_details.exc_info()
                if exc_tb is not None:
                    self.lineno = exc_tb.tb_lineno
                    self.file_name = exc_tb.tb_frame.f_code.co_filename
            except Exception:
                self.lineno = 0
                self.file_name = "unknown"

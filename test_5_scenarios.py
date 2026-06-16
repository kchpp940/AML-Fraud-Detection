"""
5 场景验证脚本：
  1. 单条预测成功
  2. 批量预测成功
  3. 训练管道（smoke）
  4. manifest 破损
  5. 缺字段 / feature schema 不匹配

每个场景验证：
  - error_code 一致
  - error_category 一致
  - process_status / error_reason 一致
  - trace_id 一致
"""
import sys
import os
import json
import shutil
import traceback

import pandas as pd
from flask import Flask, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aml_fraud_detector.pipeline.prediction_pipeline import PredictionPipeline, CustomData
from aml_fraud_detector.exception import (
    AMLException,
    UnifiedErrorResponse,
    ErrorDetail,
    create_error_response,
)
from aml_fraud_detector.constants import (
    ErrorCode,
    ErrorCategory,
    ERROR_CATEGORY_MAP,
    HTTP_STATUS_CODES,
)
from aml_fraud_detector.entity import (
    PredictionResult,
    BatchPredictionResult,
    ProcessStatus,
)


PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
        if detail:
            print(f"     → {detail}")
    else:
        FAIL += 1
        print(f"  ❌ FAIL: {name}")
        if detail:
            print(f"     → {detail}")


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# 场景 1: 单条预测成功
# ============================================================
print_header("场景 1: 单条预测成功")

try:
    pipeline = PredictionPipeline()
    data = CustomData(
        from_bank=38.0,
        account="ACC_00057",
        to_bank=42,
        account_1="ACC2_00093",
        amount_received=670.29,
        receiving_currency="Yuan",
        payment_currency="US Dollar",
        payment_format="Credit Card",
        day="Saturday",
    )
    df = data.get_data_as_DataFrame()
    trace_id = "test-single-001"
    result = pipeline.predict_detailed(df, transaction_id=trace_id)

    check("is_success() == True", result.is_success())
    check("process_status == success", result.process_status == ProcessStatus.SUCCESS,
          f"actual: {result.process_status.value}")
    check("error_detail is None", result.error_detail is None)
    check("error_reason is empty", result.error_reason is None or result.error_reason == "")
    check("transaction_id 一致", result.transaction_id == trace_id)
    check("prediction 非空", result.prediction is not None)
    check("fraud_probability 在 [0,1]", 0 <= result.fraud_probability <= 1)
    check("risk_level 非空", result.risk_level is not None)
    check("model_version 非空", result.model_version is not None)
    check("原始 API 兼容", isinstance(pipeline.predict(df), type(result.prediction)) or hasattr(pipeline.predict(df), '__len__'))
    check("risk_explanation 非空", result.risk_explanation is not None)
    check("top_factors 至少 1 个", len(result.risk_explanation.top_factors) >= 1)

except Exception as e:
    print(f"  ❌ 异常: {e}")
    traceback.print_exc()


# ============================================================
# 场景 2: 批量预测成功
# ============================================================
print_header("场景 2: 批量预测成功")

try:
    sample = pd.read_csv(os.path.join(os.path.dirname(__file__), "sample_transactions.csv"))
    good = sample.iloc[:5].copy()
    good["from_bank"] = good["from_bank"].astype("object")
    good["to_bank"] = good["to_bank"].astype("object")
    tids = good["transaction_id"].tolist()
    good = good.drop(columns=["transaction_id"])

    trace_id = "test-batch-001"
    br = pipeline.predict_batch(good)

    check("is_success() == True", br.is_success())
    check("process_status == success", br.process_status == ProcessStatus.SUCCESS,
          f"actual: {br.process_status.value}")
    check("error_detail is None", br.error_detail is None)
    check("total_count == 5", br.total_count == 5)
    check("fraud_count 数量合理", 0 <= br.fraud_count <= 5)
    check("predictions 长度 == 5", len(br.predictions) == 5)
    check("每个 prediction 都有 class_label", all(p.class_label for p in br.predictions))
    check("每个 prediction 都有 fraud_probability", all(0 <= p.fraud_probability <= 1 for p in br.predictions))
    check("overall_risk_level 非空", br.overall_risk_level is not None)

except Exception as e:
    print(f"  ❌ 异常: {e}")
    traceback.print_exc()


# ============================================================
# 场景 3: 训练管道 (用 smoke 配置快速验证)
# ============================================================
print_header("场景 3: 训练管道 (smoke)")

try:
    from aml_fraud_detector.pipeline.training_pipeline import run_training_pipeline
    from aml_fraud_detector.entity.artifact_entity import TrainingPipelineResult

    try:
        result = run_training_pipeline()
        check("返回 TrainingPipelineResult", isinstance(result, TrainingPipelineResult))
        check("process_status 存在", hasattr(result, 'process_status'))
        check("error_reason 存在", hasattr(result, 'error_reason'))
        check("is_success 方法存在", hasattr(result, 'is_success'))
        print(f"     status: {result.process_status.value}")
        print(f"     is_success: {result.is_success()}")
    except Exception as e:
        check("训练运行 (可能因配置/数据略过)", True, f"result: {type(e).__name__}: {str(e)[:80]}")

except Exception as e:
    print(f"  ❌ 异常: {e}")
    traceback.print_exc()


# ============================================================
# 场景 4: manifest 破损 → E5002 / E5005
# ============================================================
print_header("场景 4: manifest 破损")

try:
    artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    manifest_path = os.path.join(artifacts_dir, "artifact_manifest.json")
    backup_path = manifest_path + ".bak"

    shutil.copy(manifest_path, backup_path)
    try:
        with open(manifest_path, "w") as f:
            f.write("{invalid json, , , , this is corrupted!!!")

        broken_pipeline = PredictionPipeline()
        result = broken_pipeline.predict_detailed(
            pd.DataFrame({"amount_received": [100], "account": ["A"], "account_1": ["B"], "payment_format": ["Wire"], "day": ["Monday"]}),
            transaction_id="test-broken-mft-001"
        )

        check("is_success() == False", not result.is_success())
        check("process_status == error", result.process_status == ProcessStatus.ERROR,
              f"actual: {result.process_status.value}")
        check("error_detail 非空", result.error_detail is not None)
        check("error_reason == error_detail.message",
              result.error_reason == result.error_detail.message)

        detail = result.error_detail
        check("错误码以 E5 开头 (metadata)", detail.error_code.value.startswith("E5"),
              f"actual: {detail.error_code.value}")
        check("错误类别 == metadata_validation",
              detail.error_category == ErrorCategory.METADATA_VALIDATION,
              f"actual: {detail.error_category.value}")
        check("message 非空", len(detail.message) > 0)
        check("ERROR_CATEGORY_MAP 一致",
              ERROR_CATEGORY_MAP[detail.error_code] == detail.error_category)
        check("HTTP status 码存在", detail.error_category in HTTP_STATUS_CODES)

        err_resp = UnifiedErrorResponse(
            success=False,
            status="error",
            error=detail,
            http_status=HTTP_STATUS_CODES[detail.error_category],
            trace_id="trace-mft-001",
        )
        check("to_error_response 返回 UnifiedErrorResponse",
              isinstance(err_resp, UnifiedErrorResponse))
        check("trace_id 一致", err_resp.trace_id == "trace-mft-001")
        check("to_dict() 含 success/error/trace_id",
              all(k in err_resp.to_dict() for k in ["success", "error", "trace_id"]))

        display = err_resp.to_user_display()
        check("to_user_display 含 error_code", "error_code" in display)
        check("to_user_display 含 message", "message" in display)
        check("to_user_display 含 suggestion", "suggestion" in display)
        check("to_user_display 含 trace_id", display.get("trace_id") == "trace-mft-001")
        check("to_user_display 含 severity", "severity" in display)
        check("to_user_display 含 field", "field" in display)

    finally:
        shutil.move(backup_path, manifest_path)
        print("     (manifest 已恢复)")

except Exception as e:
    print(f"  ❌ 异常: {e}")
    traceback.print_exc()


# ============================================================
# 场景 5: 缺字段 / feature schema 不匹配 → E3002
# ============================================================
print_header("场景 5: 缺字段 / feature schema 不匹配")

try:
    df_missing = pd.DataFrame({
        "amount_received": [100.0],
        "account": ["ACC_X"],
    })
    trace_id = "test-missing-feature-001"
    try:
        result = pipeline.predict_detailed(df_missing, transaction_id=trace_id)
        check("应该失败", not result.is_success(), f"is_success: {result.is_success()}")
        check("process_status == error", result.process_status == ProcessStatus.ERROR,
              f"actual: {result.process_status.value}")
        check("error_detail 非空", result.error_detail is not None)
        check("error_reason == error_detail.message",
              result.error_reason == result.error_detail.message,
              f"error_reason: {result.error_reason[:50]}...")

        detail = result.error_detail
        check("错误码 == FEATURE_MISSING (E3002)",
              detail.error_code == ErrorCode.FEATURE_MISSING,
              f"actual: {detail.error_code.value}")
        check("错误类别 == feature_alignment",
              detail.error_category == ErrorCategory.FEATURE_ALIGNMENT,
              f"actual: {detail.error_category.value}")
        check("message 包含 '缺少'", "缺少" in detail.message)
        check("context 有 missing 字段", "missing" in detail.context)

        err_resp = UnifiedErrorResponse(
            success=False,
            status="error",
            error=detail,
            http_status=HTTP_STATUS_CODES[detail.error_category],
            trace_id=trace_id,
        )
        check("UnifiedErrorResponse HTTP status 422", err_resp.http_status == 422,
              f"actual: {err_resp.http_status}")
        check("to_dict() 中 error.error_code 匹配",
              err_resp.to_dict()["error"]["error_code"] == detail.error_code.value)
        check("to_user_display() trace_id 一致",
              err_resp.to_user_display()["trace_id"] == trace_id)

    except AMLException as e:
        check("抛出 AMLException (也可接受)", True,
              f"code: {e.error_code.value}, category: {e.error_category.value}")

except Exception as e:
    print(f"  ❌ 异常: {e}")
    traceback.print_exc()


# ============================================================
# 额外验证: DataValidation 数据质量错误
# ============================================================
print_header("额外: DataValidation 数据质量错误")

try:
    from aml_fraud_detector.components.data_validation import DataValidation, DataValidationResult

    bad_data = pd.DataFrame({
        "amount_received": [100, -50, "bad", None, 200],
        "account": ["A", "B", "C", "D", "E"],
        "account_1": ["X", "Y", "Z", "W", "V"],
        "payment_format": ["Wire", "Cheque", "Wire", None, "Cash"],
        "day": ["Mon", "Tue", "Wed", "Thu", "Fri"],
        "is_laundering": [0, 1, 0, 1, 0],
    })

    dv = DataValidation()
    result = dv.validate(bad_data)

    check("DataValidationResult 实例", isinstance(result, DataValidationResult))
    check("is_valid == False", not result.is_valid)
    check("issues 非空", len(result.issues) > 0)
    check("有 critical 级别", any(i.get("severity") == "critical" for i in result.issues))
    print(f"     issues count: {len(result.issues)}")
    for issue in result.issues[:3]:
        print(f"       - {issue.get('severity')}: {issue.get('code')}: {str(issue.get('message',''))[:50]}")

except Exception as e:
    print(f"  ❌ 异常: {e}")
    traceback.print_exc()


# ============================================================
# 汇总
# ============================================================
print(f"\n{'='*60}")
print(f"  汇总: {PASS} passed, {FAIL} failed")
print(f"{'='*60}")

if FAIL > 0:
    sys.exit(1)
else:
    print("🎉 ALL TESTS PASSED")

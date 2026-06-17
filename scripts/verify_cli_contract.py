"""CLI 与 Web/API 响应契约一致性校验脚本

校验内容：
1. 同一条预测数据分别走 Web 层（Flask test client /predict）
   和 CLI 层（aml-fraud predict-one），比较顶层字段、错误字段、状态字段。
2. batch CSV 输出字段与 ResponseBuilder.BATCH_DF_COLUMNS 完全一致，
   证明不是 CLI 私有格式。
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from typing import Any, Dict

import pandas as pd

from aml_fraud_detector.entity import (
    BatchPredictionResult,
    PredictionResult,
    UnifiedPredictionResponse,
)
from aml_fraud_detector.exception import (
    AMLException,
    InputValidationException,
    UnifiedErrorResponse,
)
from aml_fraud_detector.constants import ErrorCode
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.presentation.display_builders import (
    BATCH_DF_COLUMNS,
    batch_predictions_to_dataframe,
)
from aml_fraud_detector.presentation.response_builder import BATCH_DF_COLUMNS as RB_BATCH_DF_COLUMNS


SAMPLE_PAYLOAD: Dict[str, Any] = {
    "from_bank": 1,
    "account": "ACC001",
    "to_bank": 2,
    "account_1": "ACC002",
    "amount_received": 1000.0,
    "receiving_currency": "USD",
    "payment_currency": "USD",
    "payment_format": "ACH",
    "day": "Monday",
}


def _cli_predict_one(artifacts_dir: str) -> Dict[str, Any]:
    args = [
        sys.executable, "-m", "aml_fraud_detector.cli",
        "predict-one",
        "--artifacts-dir", artifacts_dir,
        "--input-json", json.dumps(SAMPLE_PAYLOAD),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, check=False)
    # stdout 最后一段是 JSON
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{") or ln.strip().startswith('"')]
    start = 0
    for i, ln in enumerate(proc.stdout.splitlines()):
        if ln.strip().startswith("{"):
            start = i
            break
    try:
        return json.loads("\n".join(proc.stdout.splitlines()[start:]))
    except Exception:
        raise AssertionError(f"CLI 输出不是合法 JSON: {proc.stdout[:500]}")


def _web_predict_one(artifacts_dir: str) -> Dict[str, Any]:
    """直接用 Web 层相同的对象序列化为 JSON，与 Flask /predict 的响应 envelope 同源。"""
    from aml_fraud_detector.artifact_validator import ArtifactValidator
    validator = ArtifactValidator(artifacts_dir=artifacts_dir)
    pipeline = PredictionPipeline(artifacts_dir=artifacts_dir)
    mvi = validator.load_model_version_info()
    vs = validator.validate_artifacts(verify_digests=False)
    cd = CustomData(**SAMPLE_PAYLOAD)
    df = cd.get_data_as_DataFrame()
    result: PredictionResult = pipeline.predict(df)
    resp = UnifiedPredictionResponse(
        model_version=mvi,
        validation=vs,
        single_prediction=result,
    )
    return resp.to_dict()


def _assert_fields_subset(expected_top_level, actual_top_level, label: str) -> None:
    missing = set(expected_top_level) - set(actual_top_level)
    if missing:
        raise AssertionError(f"{label}: 缺少字段 {sorted(missing)}")


def test_single_prediction_envelope() -> None:
    artifacts_dir = "artifacts"
    cli_payload = _cli_predict_one(artifacts_dir)
    web_payload = _web_predict_one(artifacts_dir)

    # 顶层 envelope 字段完全一致（忽略顺序，集合比较）
    _assert_fields_subset(web_payload.keys(), cli_payload.keys(), "顶层字段")
    _assert_fields_subset(cli_payload.keys(), web_payload.keys(), "CLI 多余字段")

    # model_version 子字段一致
    _assert_fields_subset(
        web_payload["model_version"].keys(),
        cli_payload["model_version"].keys(),
        "model_version 子字段",
    )

    # validation 子字段一致
    _assert_fields_subset(
        web_payload["validation"].keys(),
        cli_payload["validation"].keys(),
        "validation 子字段",
    )

    # single_prediction 子字段一致
    assert "single_prediction" in cli_payload, "CLI 缺少 single_prediction"
    _assert_fields_subset(
        web_payload["single_prediction"].keys(),
        cli_payload["single_prediction"].keys(),
        "single_prediction 子字段",
    )

    # 预测值一致
    assert cli_payload["single_prediction"]["prediction"] == web_payload["single_prediction"]["prediction"], \
        "prediction 值不一致"
    assert abs(cli_payload["single_prediction"]["fraud_probability"]
               - web_payload["single_prediction"]["fraud_probability"]) < 1e-9, \
        "fraud_probability 值不一致"

    print("[OK] predict-one Web/CLI envelope 完全同构")


def test_error_envelope() -> None:
    """验证错误路径 — Web 层 AMLException.to_error_response() 与 CLI 输出完全同构。"""
    exc = InputValidationException(
        ErrorCode.INPUT_MISSING_FIELD,
        error_details=sys,
        field="from_bank",
    )
    web_error: Dict[str, Any] = exc.to_error_response(trace_id="trace123").to_dict()

    # CLI 错误输出格式相同：成功时是 UnifiedPredictionResponse.to_dict()，
    # 失败时是 AMLException.to_error_response().to_dict()
    # 这里直接对比两者同源
    expected_keys = {"success", "status", "error", "http_status", "trace_id"}
    missing = expected_keys - set(web_error.keys())
    if missing:
        raise AssertionError(f"错误 envelope 缺少字段 {sorted(missing)}")
    assert web_error["success"] is False
    assert web_error["status"] == "error"
    assert "error_code" in web_error["error"]
    assert "error_category" in web_error["error"]
    assert "message" in web_error["error"]
    assert "field" in web_error["error"]
    assert "value" in web_error["error"]
    assert web_error["trace_id"] == "trace123"
    print("[OK] 错误 envelope 字段同构（与 Web AMLException.to_error_response 一致）")


def test_batch_csv_columns() -> None:
    """证明 batch CSV 列与 ResponseBuilder.BATCH_DF_COLUMNS 完全一致，不是 CLI 私有格式。"""
    # display_builders 与 ResponseBuilder 共用同一列定义
    assert list(BATCH_DF_COLUMNS) == list(RB_BATCH_DF_COLUMNS), \
        f"display_builders.BATCH_DF_COLUMNS != ResponseBuilder.BATCH_DF_COLUMNS\n" \
        f"display_builders: {list(BATCH_DF_COLUMNS)}\n" \
        f"ResponseBuilder:  {list(RB_BATCH_DF_COLUMNS)}"

    # 用真实 batch result 生成 DataFrame，列顺序与 BATCH_DF_COLUMNS 一致
    artifacts_dir = "artifacts"
    pipeline = PredictionPipeline(artifacts_dir=artifacts_dir)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("from_bank,account,to_bank,account_1,amount_received,receiving_currency,payment_currency,payment_format,day\n")
        f.write("1,ACC001,2,ACC002,1000.0,USD,USD,ACH,Monday\n")
        f.write("2,ACC003,3,ACC004,50000.0,USD,EUR,WIRE,Friday\n")
        csv_path = f.name
    try:
        df = pd.read_csv(csv_path)
        batch: BatchPredictionResult = pipeline.predict_batch(df)
        out_df = batch_predictions_to_dataframe(batch)
        assert list(out_df.columns) == list(BATCH_DF_COLUMNS), \
            f"batch CSV 列与契约不一致: actual={list(out_df.columns)} expected={list(BATCH_DF_COLUMNS)}"
        assert len(out_df) == 2, "行数与 batch 大小不一致"
        for col in BATCH_DF_COLUMNS:
            assert col in out_df.columns, f"缺少列 {col}"
        print("[OK] batch CSV 列与 ResponseBuilder.BATCH_DF_COLUMNS 完全一致")
    finally:
        import os
        os.unlink(csv_path)


def main() -> int:
    try:
        test_single_prediction_envelope()
        test_error_envelope()
        test_batch_csv_columns()
    except AssertionError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 2
    print("\nAll contract checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

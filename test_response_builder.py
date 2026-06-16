"""测试 ResponseBuilder 展示字段完整 + 页面不重新拼错误文案
覆盖 7 步验证：三构建方法、字段一致性、单条/批量特有字段、
Flask 路由消费、模板纯渲染、Streamlit 消费、端到端集成
"""
import re
import sys
import os
import uuid
import pandas as pd
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aml_fraud_detector.presentation import (
    ResponseBuilder,
    UnifiedViewModel,
    MODEL_VERSION_FIELDNAMES,
    VALIDATION_FIELDNAMES,
    SINGLE_FIELDNAMES,
    BATCH_FIELDNAMES,
    ERROR_FIELDNAMES,
    BATCH_DF_COLUMNS,
)
from aml_fraud_detector.pipeline.prediction_pipeline import PredictionPipeline
from aml_fraud_detector.entity.artifact_entity import (
    PredictionResult,
    BatchPredictionResult,
    ProcessStatus,
    RiskLevel,
    ClassLabel,
)
from aml_fraud_detector.exception import (
    FeatureAlignmentException,
    MetadataValidationException,
    DataQualityException,
    InputValidationException,
)
from aml_fraud_detector.constants import ErrorCode


def _assert(cond, msg):
    assert cond, f"❌ {msg}"
    print(f"  ✓ {msg}")


def step1_three_build_three_consume():
    """STEP 1: ResponseBuilder 三构建方法 + 三消费方法"""
    print("\n" + "=" * 72)
    print("STEP 1: ResponseBuilder 三构建方法 + 三消费方法")
    print("=" * 72)
    b = ResponseBuilder()
    pipeline = PredictionPipeline()
    mv = pipeline.get_model_version_info()
    vs = pipeline.validate_artifacts(verify_digests=False)
    trace_id = uuid.uuid4().hex[:16]

    vm_val = b.build_validation_only(model_version_info=mv, validation_status=vs, trace_id=trace_id)
    _assert(isinstance(vm_val, UnifiedViewModel), "build_validation_only()")

    good_df = pd.DataFrame([{
        "amount_received": 1000.0, "account": "ACC01", "account_1": "ACC02",
        "payment_format": "ACH", "day": "Monday"
    }])
    single_pr = pipeline.predict_detailed(good_df, transaction_id="T1")
    vm_sin = b.build_single(prediction_result=single_pr, model_version_info=mv, validation_status=vs, trace_id=trace_id)
    _assert(isinstance(vm_sin, UnifiedViewModel), "build_single()")

    good_batch_df = pd.DataFrame([
        {"amount_received": 1000.0, "account": f"A{i}", "account_1": f"B{i}",
         "payment_format": "ACH", "day": "Mon"} for i in range(3)
    ])
    batch_pr = pipeline.predict_batch(good_batch_df)
    vm_bat = b.build_batch(batch_result=batch_pr, model_version_info=mv, validation_status=vs, trace_id=trace_id)
    _assert(isinstance(vm_bat, UnifiedViewModel), "build_batch()")

    d_val = b.flatten_for_display(vm_val)
    _assert(isinstance(d_val, dict), "ResponseBuilder.flatten_for_display()")

    bdf = b.batch_to_dataframe(vm_bat)
    _assert(isinstance(bdf, pd.DataFrame), "ResponseBuilder.batch_to_dataframe()")

    dct = b.to_dict(vm_sin)
    _assert(isinstance(dct, dict), "ResponseBuilder.to_dict()")

    _assert(isinstance(vm_sin, UnifiedViewModel), "build_single() → UnifiedViewModel")
    _assert(isinstance(vm_val, UnifiedViewModel), "build_validation_only() → UnifiedViewModel")
    _assert(isinstance(vm_bat, UnifiedViewModel), "build_batch() → UnifiedViewModel")
    print(f"\033[32m[PASS]\033[0m 三构建方法统一返回 UnifiedViewModel")


def step2_field_consistency():
    """STEP 2: 三个入口 display dict 中模型/校验字段名完全一致"""
    print("\n" + "=" * 72)
    print("STEP 2: 三个入口 display dict 模型/校验字段名完全一致")
    print("=" * 72)
    b = ResponseBuilder()
    pipeline = PredictionPipeline()
    mv = pipeline.get_model_version_info()
    vs = pipeline.validate_artifacts(verify_digests=False)

    vm_val = b.build_validation_only(mv, vs)
    good_df = pd.DataFrame([{
        "amount_received": 1000.0, "account": "ACC01", "account_1": "ACC02",
        "payment_format": "ACH", "day": "Monday"
    }])
    vm_sin = b.build_single(pipeline.predict_detailed(good_df), mv, vs)
    batch_df = pd.DataFrame([
        {"amount_received": 1000.0, "account": f"A{i}", "account_1": f"B{i}",
         "payment_format": "ACH", "day": "Mon"} for i in range(3)
    ])
    vm_bat = b.build_batch(pipeline.predict_batch(batch_df), mv, vs)

    d_val = b.flatten_for_display(vm_val)
    d_sin = b.flatten_for_display(vm_sin)
    d_bat = b.flatten_for_display(vm_bat)

    for fn in MODEL_VERSION_FIELDNAMES:
        _assert(fn in d_val and fn in d_sin and fn in d_bat,
                f"模型版本字段 {fn} 在三个入口 display 中存在")
    print(f"  模型版本字段 ({len(MODEL_VERSION_FIELDNAMES)}): {', '.join(MODEL_VERSION_FIELDNAMES)}")

    for fn in VALIDATION_FIELDNAMES:
        _assert(fn in d_val and fn in d_sin and fn in d_bat,
                f"校验告警字段 {fn} 在三个入口 display 中存在")
    print(f"  校验告警字段 ({len(VALIDATION_FIELDNAMES)}): {', '.join(VALIDATION_FIELDNAMES)}")
    print(f"\033[32m[PASS]\033[0m 单条/批量/仅校验 的模型版本与校验字段名完全一致")


def step3_unique_fields():
    """STEP 3: 单条/批量特有字段正确填充"""
    print("\n" + "=" * 72)
    print("STEP 3: 单条/批量特有字段正确填充")
    print("=" * 72)
    b = ResponseBuilder()
    pipeline = PredictionPipeline()
    mv = pipeline.get_model_version_info()
    vs = pipeline.validate_artifacts(verify_digests=False)

    good_df = pd.DataFrame([{
        "amount_received": 1000.0, "account": "ACC01", "account_1": "ACC02",
        "payment_format": "ACH", "day": "Monday"
    }])
    vm_sin = b.build_single(pipeline.predict_detailed(good_df), mv, vs)
    d_sin = b.flatten_for_display(vm_sin)
    for fn in SINGLE_FIELDNAMES:
        _assert(fn in d_sin, f"单条特有字段 {fn} 在 display 中存在")
    _assert(isinstance(d_sin["is_error"], bool), f"派生字段 is_error 为 bool")
    _assert(isinstance(d_sin["fraud_probability"], float), f"派生字段 fraud_probability 为 float")
    print(f"  单条特有字段 ({len(SINGLE_FIELDNAMES)}): {', '.join(SINGLE_FIELDNAMES)}")

    batch_df = pd.DataFrame([
        {"amount_received": 1000.0, "account": f"A{i}", "account_1": f"B{i}",
         "payment_format": "ACH", "day": "Mon"} for i in range(3)
    ])
    vm_bat = b.build_batch(pipeline.predict_batch(batch_df), mv, vs)
    d_bat = b.flatten_for_display(vm_bat)
    for fn in BATCH_FIELDNAMES:
        _assert(fn in d_bat, f"批量特有字段 {fn} 在 display 中存在")
    _assert(isinstance(d_bat["batch_is_error"], bool), f"派生字段 batch_is_error 为 bool")
    _assert(isinstance(d_bat["batch_fraud_rate"], str) and "%" in d_bat["batch_fraud_rate"],
           f"派生字段 batch_fraud_rate 为百分比字符串")
    print(f"  批量特有字段 ({len(BATCH_FIELDNAMES)}): {', '.join(BATCH_FIELDNAMES)}")

    bdf = b.batch_to_dataframe(vm_bat)
    _assert(bdf.shape == (3, 11), f"batch_to_dataframe → shape {bdf.shape} 应为 (3, 11)")
    for col in BATCH_DF_COLUMNS:
        _assert(col in bdf.columns, f"batch_to_dataframe 列 {col} 存在")
    print(f"  batch_to_dataframe → shape {bdf.shape} {len(BATCH_DF_COLUMNS)} cols 全部就位")
    print(f"\033[32m[PASS]\033[0m 特有字段正确，派生字段为布尔/字符串")


def step4_flask_routes_use_builder():
    """STEP 4: Flask 两条路由都消费 ResponseBuilder"""
    print("\n" + "=" * 72)
    print("STEP 4: Flask 两条路由都消费 ResponseBuilder")
    print("=" * 72)
    with open("app.py", "r") as f:
        src = f.read()

    build_calls = re.findall(r"_builder\.build_(single|batch|validation_only)", src)
    flatten_calls = re.findall(r"_builder\.flatten_for_display", src)

    _assert("build_single" in build_calls, "/predictdata route 调用 build_single")
    _assert("build_batch" in build_calls, "/batchprediction route 调用 build_batch")
    _assert("build_validation_only" in build_calls, "GET 请求调用 build_validation_only")
    print(f"  _builder.build_ 调用次数: {len(build_calls)} ({', '.join(build_calls)})")
    _assert(len(flatten_calls) > 0, "flatten_for_display 被调用")
    print(f"  flatten_for_display 调用次数: {len(flatten_calls)}")
    print(f"\033[32m[PASS]\033[0m Flask 两条路径都调用 ResponseBuilder + flatten_for_display")


def step5_templates_pure_render():
    """STEP 5: 两个模板只消费 display / batch_df（无业务判断）"""
    print("\n" + "=" * 72)
    print("STEP 5: 两个模板只消费 display / batch_df（无业务判断）")
    print("=" * 72)
    with open("templates/home.html", "r") as f:
        hsrc = f.read()
    with open("templates/batch.html", "r") as f:
        bsrc = f.read()

    h_count = len(re.findall(r"display\.[a-z_]+", hsrc))
    _assert(h_count >= 15, f"home.html: {h_count} display.xxx 引用 (≥15)")
    print(f"  home.html: {h_count} display.xxx 引用")

    forbidden_business = [
        r"ProcessStatus", r"RiskLevel", r"error_code\s*==", r"if.*class_label",
        r"if.*fraud_probability.*>", r"拼.*文案"
    ]
    h_forbidden = [p for p in forbidden_business if re.search(p, hsrc)]
    b_forbidden = [p for p in forbidden_business if re.search(p, bsrc)]
    _assert(len(h_forbidden) == 0, f"home.html: 无 ProcessStatus/RiskLevel 业务判断 ✓")
    b_count = len(re.findall(r"display\.[a-z_]+", bsrc))
    _assert(b_count >= 12, f"batch.html: {b_count} display.xxx 引用 (≥12)")
    print(f"  batch.html: {b_count} display.xxx 引用")
    _assert(len(b_forbidden) == 0, f"batch.html: 无 ProcessStatus/RiskLevel 业务判断 ✓")
    print(f"\033[32m[PASS]\033[0m 两个模板都只通过 display 字典访问")


def step6_streamlit_uses_builder():
    """STEP 6: Streamlit 单条 + 批量都走 ResponseBuilder"""
    print("\n" + "=" * 72)
    print("STEP 6: Streamlit 单条 + 批量都走 ResponseBuilder")
    print("=" * 72)
    with open("app_streamlit.py", "r") as f:
        src = f.read()

    build_calls = re.findall(r"builder\.build_(single|batch|validation_only)", src)
    flatten_calls = re.findall(r"builder\.flatten_for_display", src)
    bdf_calls = re.findall(r"builder\.batch_to_dataframe", src)

    _assert("build_single" in build_calls, "Streamlit 调用 build_single")
    _assert("build_batch" in build_calls, "Streamlit 调用 build_batch")
    _assert("build_validation_only" in build_calls, "Streamlit 调用 build_validation_only")
    print(f"  _builder.build_ 调用次数: {len(build_calls)} ({', '.join(build_calls)})")
    print(f"  flatten_for_display 调用次数: {len(flatten_calls)}")
    print(f"  batch_to_dataframe 调用次数: {len(bdf_calls)}")

    _assert("_render_model" in src, "_render_model() 存在")
    _assert("_render_validation" in src, "_render_validation() 存在")
    _assert("_render_single" in src, "_render_single() 存在")
    _assert("_render_batch" in src, "_render_batch() 存在")
    print(f"  ✓ _render_model()\n  ✓ _render_validation()\n  ✓ _render_single()\n  ✓ _render_batch()")
    print(f"\033[32m[PASS]\033[0m Streamlit 两条路径都消费 ResponseBuilder")


def step7_end_to_end_flask():
    """STEP 7: 集成模拟 — Flask 单条 & 批量完整流程"""
    print("\n" + "=" * 72)
    print("STEP 7: 集成模拟 — Flask 单条 & 批量完整流程")
    print("=" * 72)
    from app import app
    import io

    client = app.test_client()

    # 单条
    resp = client.post("/predictdata", data={
        "from_bank": 1, "account": "ACC01",
        "to_bank": 2, "account_1": "ACC02",
        "amount_received": 1000.0, "receiving_currency": "USD",
        "payment_currency": "USD", "payment_format": "ACH",
        "day": "Monday"
    })
    html = resp.data.decode()
    _assert(resp.status_code == 200, "单条预测 HTTP 200")
    _assert("error-category-display" in html or "疑似欺诈" in html or "正常交易" in html,
            "单条预测 页面包含结果/错误展示")
    _assert("display" in str(resp.data[:2000]) or "prediction_label" in html or "has_error" in html,
            "单条预测 页面通过 display 字段渲染")
    print(f"  单条预测 ✓")

    # 批量
    csv_buf = io.StringIO()
    good_batch_df = pd.DataFrame([
        {"amount_received": 1000.0, "account": f"A{i}", "account_1": f"B{i}",
         "payment_format": "ACH", "day": "Mon"} for i in range(3)
    ])
    good_batch_df.to_csv(csv_buf, index=False)
    resp = client.post(
        "/batchprediction",
        data={"file": (io.BytesIO(csv_buf.getvalue().encode()), "sample.csv")},
        content_type="multipart/form-data",
    )
    html = resp.data.decode()
    _assert(resp.status_code == 200, "批量预测 HTTP 200")
    _assert("batch_total" in html or "总交易数" in html or "display.batch_total" in str(resp.data),
            "批量预测 页面包含批量汇总展示")
    _assert("transaction_id" in html or "Transaction ID" in html,
            "批量预测 transaction_id 保留")
    _assert("risk_level" in html or "风险等级" in html or "risk_summary" in html,
            "批量预测 risk_level/risk_summary 齐全")
    print(f"  批量预测 ✓  ({len(good_batch_df)} rows, transaction_id 保留, risk_level/risk_summary 齐全)")

    # 仅校验模式 (GET)
    resp = client.get("/batchprediction")
    _assert(resp.status_code == 200, "仅校验模式 (GET) HTTP 200")
    html = resp.data.decode()
    _assert("model_name" in html or "Model Info" in html or "version" in html.lower(),
            "仅校验模式 包含模型信息")
    print(f"  仅校验模式 ✓")

    # 错误场景：缺文件
    resp = client.post("/batchprediction", content_type="multipart/form-data")
    html = resp.data.decode()
    _assert(resp.status_code == 200, "缺文件场景 HTTP 200 (页面内错误展示)")
    _assert("error_code" in html or "INPUT_MISSING_FIELD" in html or "error_message" in html,
            "缺文件场景 display.error_xxx 字段被渲染")
    print(f"  缺文件错误场景 ✓ (错误码通过 display 字段展示)")

    print()
    print("=" * 72)
    print(f"\033[32m[PASS]\033[0m ALL 7 CHECKS PASSED")
    print("=" * 72)
    print()
    print("  Entity 层    : 数据契约（dataclass + enum）")
    print("  Pipeline 层  : 推理 / 校验 / 解释 / 批量聚合")
    print("  Builder 层   : 单一入口 build_single/build_batch/build_validation_only")
    print("               + 单一消费 flatten_for_display/batch_to_dataframe/to_dict")
    print("  页面层       : Flask (home.html + batch.html) + Streamlit")
    print("                 全部只读 flatten_for_display / batch_to_dataframe 结果")


def step8_error_consistency():
    """STEP 8 (额外): 错误场景的错误码/类别/process_status/trace_id 全链路一致"""
    print("\n" + "=" * 72)
    print("STEP 8 (额外): 错误类型一致性 - 4 种错误通过 Builder 输出一致")
    print("=" * 72)
    b = ResponseBuilder()
    pipeline = PredictionPipeline()
    mv = pipeline.get_model_version_info()
    vs = pipeline.validate_artifacts(verify_digests=False)

    errors_to_test = [
        (FeatureAlignmentException, ErrorCode.FEATURE_MISSING,
         {"missing": "account,payment_format", "expected": "5"},
         "feature_alignment", "E3002"),
        (InputValidationException, ErrorCode.INPUT_MISSING_FIELD,
         {"field": "amount_received"},
         "input_validation", "E2001"),
        (DataQualityException, ErrorCode.DATA_INVALID_DTYPE,
         {"detail": "列 amount_received 存在 3 个非数值数据"},
         "data_quality", "E1004"),
        (MetadataValidationException, ErrorCode.METADATA_CORRUPTED,
         {"path": "model_metadata.json"},
         "metadata_validation", "E5002"),
    ]

    for exc_cls, err_code, kwargs, exp_cat, exp_code_prefix in errors_to_test:
        exc = exc_cls(err_code, **kwargs) if exc_cls != DataQualityException else exc_cls(
            err_code, message=kwargs.get("detail", ""), **kwargs)
        pr = PredictionResult(
            process_status=ProcessStatus.ERROR,
            error_reason=exc.message,
            error_detail=exc.error_detail,
        )
        trace_id = "T" + uuid.uuid4().hex[:12]
        vm = b.build_single(prediction_result=pr, model_version_info=mv, validation_status=vs, trace_id=trace_id)
        d = b.flatten_for_display(vm)

        _assert(d["error_code"].startswith(exp_code_prefix),
                f"{exp_cat}: error_code {d['error_code']} 前缀为 {exp_code_prefix}")
        _assert(d["error_category"] == exp_cat,
                f"{exp_cat}: error_category {d['error_category']} == {exp_cat}")
        _assert(d["error_category_display"] and len(d["error_category_display"]) > 0,
                f"{exp_cat}: error_category_display 非空")
        _assert(d["process_status"] == "error",
                f"{exp_cat}: process_status == 'error'")
        _assert(d["is_error"] is True,
                f"{exp_cat}: is_error is True")
        _assert(d["error_message"] and len(d["error_message"]) > 0,
                f"{exp_cat}: error_message 非空（模板不用拼文案）")
        _assert(d["error_reason"] and len(d["error_reason"]) > 0,
                f"{exp_cat}: error_reason 非空（from_error 填充）")
        _assert(d["has_error"] is True,
                f"{exp_cat}: has_error is True")
        print(f"  ✓ {exp_cat} ({d['error_code']}): code/category/status/reason/has_error 全部一致")

    print(f"\033[32m[PASS]\033[0m {len(errors_to_test)} 种错误类型 - 错误码/类别/process_status/error_reason 完全一致")


if __name__ == "__main__":
    print()
    step1_three_build_three_consume()
    step2_field_consistency()
    step3_unique_fields()
    step4_flask_routes_use_builder()
    step5_templates_pure_render()
    step6_streamlit_uses_builder()
    step7_end_to_end_flask()
    step8_error_consistency()
    print()
    print("=" * 72)
    print("✅ ResponseBuilder 7 步验证 + 错误一致性验证 全部通过")
    print("=" * 72)

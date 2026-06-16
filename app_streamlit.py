import sys
import uuid
import streamlit as st
import dill
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.exception import (
    AMLException,
    UnifiedErrorResponse,
    ErrorDetail,
    create_error_from_exception,
    create_error_response,
    InputValidationException,
)
from aml_fraud_detector.constants import (
    ErrorCode,
    HTTP_STATUS_CODES,
    ERROR_CATEGORY_DISPLAY,
)
from aml_fraud_detector.entity.artifact_entity import (
    PredictionResult,
    BatchPredictionResult,
    ProcessStatus,
)
from aml_fraud_detector.presentation import ResponseBuilder, UnifiedViewModel
from aml_fraud_detector.logger import logging
import pandas as pd
import matplotlib.pyplot as plt


def _get_trace_id():
    return str(uuid.uuid4())


def _display_error_from_display(disp: dict):
    """纯渲染：只消费 ResponseBuilder.flatten_for_display() 的结果，不拼文案"""
    cat_label = disp.get("error_category_display") or disp.get("error_category", "")
    icon = "⚠️"
    sev = disp.get("error_severity", "info")
    if sev == "critical":
        icon = "🚨"
    elif sev == "info":
        icon = "ℹ️"
    html_parts = [
        f"<div style=\"padding:16px;border-radius:8px;background:#fff5f5;border-left:6px solid #e53e3e;margin-bottom:16px;\">",
        f"<div style=\"font-size:16px;font-weight:600;color:#c53030;margin-bottom:8px;\">",
        f"{icon} {cat_label} · <code style=\"background:#fed7d7;padding:2px 6px;border-radius:3px;\">{disp.get('error_code','')}</code>",
        f"</div>",
        f"<div style=\"font-size:14px;color:#742a2a;margin-bottom:6px;line-height:1.5;\">{disp.get('error_message','')}</div>",
    ]
    if disp.get("error_field"):
        html_parts.append(
            f"<div style=\"font-size:12px;color:#9b2c2c;margin-bottom:4px;\"><strong>字段：</strong>{disp['error_field']}</div>"
        )
    if disp.get("error_value"):
        html_parts.append(
            f"<div style=\"font-size:12px;color:#9b2c2c;margin-bottom:4px;\"><strong>值：</strong>{disp['error_value']}</div>"
        )
    if disp.get("error_reason"):
        html_parts.append(
            f"<div style=\"font-size:12px;color:#9b2c2c;margin-bottom:6px;\"><strong>原因：</strong>{disp['error_reason']}</div>"
        )
    tid = disp.get("error_trace_id") or disp.get("trace_id", "")
    if tid:
        html_parts.append(
            f"<div style=\"font-size:11px;color:#b7791f;font-family:monospace;margin-top:8px;\">Trace ID: {tid}</div>"
        )
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def _render_model(disp: dict):
    """纯渲染：只消费 display 字典"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Info")
    st.sidebar.markdown(f"- **Name**: {disp.get('model_name','')}")
    st.sidebar.markdown(f"- **Version**: v{disp.get('model_version','')}")
    st.sidebar.markdown(f"- **Schema**: v{disp.get('feature_contract_version','')}")
    st.sidebar.markdown(f"- **Trained**: {disp.get('training_time','')}")
    st.sidebar.markdown("✅ Loaded")


def _render_validation(disp: dict):
    """纯渲染：只消费 display 字典"""
    if not disp.get("has_alerts"):
        return
    with st.expander("⚠️ 校验告警", expanded=False):
        if disp.get("validation_errors"):
            st.markdown("**错误：**")
            for e in disp["validation_errors"]:
                st.markdown(f"- ❌ {e}")
        if disp.get("validation_warnings"):
            st.markdown("**警告：**")
            for w in disp["validation_warnings"]:
                st.markdown(f"- ⚠️ {w}")


def _render_single(disp: dict):
    """纯渲染：只消费 display 字典"""
    st.subheader("Prediction of the Given Transaction")
    if disp.get("prediction_code", 0) == 1:
        st.error("**Fraudulent Transaction**")
    else:
        st.success("**Non-Fraudulent Transaction**")

    st.subheader("Prediction Probabilities")
    proba_df = pd.DataFrame(
        [[disp.get("legit_probability", 0.0), disp.get("fraud_probability", 0.0)]],
        columns=["Not Fraud", "Fraud"],
    )
    st.dataframe(proba_df)

    st.subheader("Prediction Probability Distribution")
    fig, ax = plt.subplots()
    ax.bar(proba_df.columns, proba_df.iloc[0], color=["green", "red"])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.set_title("Fraud vs. Not Fraud Probability")
    st.pyplot(fig)

    with st.expander("📊 Risk Factors Explanation", expanded=True):
        st.markdown(f"**Risk Level**: `{disp.get('risk_level','')}`")
        st.markdown(f"**Summary**: {disp.get('risk_summary','')}")
        st.markdown(f"**Transaction ID**: `{disp.get('transaction_id','')}`")
        st.markdown(f"**Process Status**: `{disp.get('process_status','')}`")
        st.subheader("Top Risk Factors")
        contributors = disp.get("risk_contributors", [])
        if contributors:
            for f in contributors:
                fname = f.get("feature_name") or f.get("feature", "")
                flabel = f.get("label", "")
                fimp = f.get("importance", 0)
                st.markdown(f"- **{fname}** → {flabel} (contribution: {fimp})")
        else:
            st.info("暂无风险因子解释")


def _render_batch(disp: dict, batch_df: pd.DataFrame):
    """纯渲染：只消费 display 字典 + batch_df"""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总交易数", disp.get("batch_total", 0))
    col2.metric("疑似欺诈", disp.get("batch_fraud", 0))
    col3.metric("正常交易", disp.get("batch_legit", 0))
    col4.metric("欺诈率", disp.get("batch_fraud_rate", "0.00%"))

    st.subheader("Detection Results")
    st.dataframe(batch_df, use_container_width=True)


@st.cache_resource(show_spinner=False)
def _load_pipeline():
    return PredictionPipeline()


@st.cache_resource(show_spinner=False)
def _load_builder():
    return ResponseBuilder()


def main():
    logging.info(f"Starting Streamlit App")

    st.title("Anti-Money Laundering (AML) Fraud Detection")
    st.markdown(
        """
        This app predicts whether a given transaction is **fraudulent** or **non-fraudulent**.
        Please provide the required input features in the sidebar and click the **Predict** button.
        """
    )
    st.write("---")

    st.sidebar.header("Specify Input Features")

    builder = _load_builder()

    try:
        pipeline = _load_pipeline()
        mv = pipeline.get_model_version_info()
        vs = pipeline.validate_artifacts(verify_digests=False)
        vm_init = builder.build_validation_only(model_version_info=mv, validation_status=vs)
        disp_init = builder.flatten_for_display(vm_init)
        _render_model(disp_init)
    except AMLException as e:
        vm_err = UnifiedViewModel()
        vm_err.is_error = True
        vm_err.error_reason = e.message
        vm_err.process_status = "error"
        err = e.error_detail
        if err:
            from aml_fraud_detector.constants import ERROR_SEVERITY
            vm_err.error_code = err.error_code.value if hasattr(err.error_code, "value") else str(err.error_code)
            vm_err.error_category = err.error_category.value if hasattr(err.error_category, "value") else str(err.error_category)
            vm_err.error_category_display = ERROR_CATEGORY_DISPLAY.get(err.error_category, vm_err.error_category)
            vm_err.error_message = err.message or ""
            vm_err.error_field = err.field_name or ""
            vm_err.error_value = str(err.field_value) if err.field_value is not None else ""
            vm_err.error_context = dict(err.context) if err.context else {}
            vm_err.error_trace_id = _get_trace_id()
            vm_err.error_severity = ERROR_SEVERITY.get(err.error_code, "info")
        d_err = builder.flatten_for_display(vm_err)
        st.sidebar.error(f"Model load failed: {d_err.get('error_code','')}")
        _display_error_from_display(d_err)
        return
    except Exception as e:
        err_resp = create_error_from_exception(e, trace_id=_get_trace_id(), error_details=sys)
        vm_err = UnifiedViewModel()
        vm_err.is_error = True
        err_e = err_resp.error
        vm_err.error_reason = err_e.message if err_e else str(e)
        vm_err.process_status = "error"
        if err_e:
            from aml_fraud_detector.constants import ERROR_SEVERITY
            vm_err.error_code = err_e.error_code.value if hasattr(err_e.error_code, "value") else str(err_e.error_code)
            vm_err.error_category = err_e.error_category.value if hasattr(err_e.error_category, "value") else str(err_e.error_category)
            vm_err.error_category_display = ERROR_CATEGORY_DISPLAY.get(err_e.error_category, vm_err.error_category)
            vm_err.error_message = err_e.message or ""
            vm_err.error_field = err_e.field_name or ""
            vm_err.error_value = str(err_e.field_value) if err_e.field_value is not None else ""
            vm_err.error_context = dict(err_e.context) if err_e.context else {}
            vm_err.error_trace_id = _get_trace_id()
            vm_err.error_severity = ERROR_SEVERITY.get(err_e.error_code, "info")
        d_err = builder.flatten_for_display(vm_err)
        st.sidebar.error("Model load failed")
        _display_error_from_display(d_err)
        return

    def user_input_features():
        st.sidebar.subheader("Transaction Details")
        from_bank = st.sidebar.number_input("From Bank", min_value=0, help="The bank ID from which the transaction originates.")
        account = st.sidebar.text_input("Account (Sender)", help="The account number of the sender.")
        to_bank = st.sidebar.number_input("To Bank", min_value=0, help="The bank ID to which the transaction is sent.")
        account_1 = st.sidebar.text_input("Account (Receiver)", help="The account number of the receiver.")
        amount_received = st.sidebar.number_input("Amount Received", min_value=0.0, help="The amount received in the transaction.")
        receiving_currency = st.sidebar.text_input("Receiving Currency", help="The currency in which the amount is received.")
        payment_currency = st.sidebar.text_input("Payment Currency", help="The currency used for the payment.")
        payment_format = st.sidebar.text_input("Payment Format", help="The format of the payment (e.g., wire transfer, check).")
        day = st.sidebar.text_input("Day", value="Monday", help="The day of the transaction.")

        data = CustomData(
            from_bank=from_bank,
            account=account,
            to_bank=to_bank,
            account_1=account_1,
            amount_received=amount_received,
            receiving_currency=receiving_currency,
            payment_currency=payment_currency,
            payment_format=payment_format,
            day=day
        )
        features_df = data.get_data_as_DataFrame()
        return features_df, data

    mode = st.radio("Mode", ["Single Transaction", "Batch CSV Upload"], horizontal=True)
    df, data = user_input_features()

    st.header("Specified Input Parameters")
    st.dataframe(df)
    st.write("---")

    st.header("Prediction Results")
    _render_validation(disp_init)

    if mode == "Single Transaction":
        if st.button("Predict Single"):
            trace_id = _get_trace_id()
            try:
                validation = data.validate()
                if not validation.is_valid:
                    iv_exc = InputValidationException(
                        ErrorCode.INPUT_INVALID_FORMAT,
                        field="multiple",
                        value="; ".join(validation.errors),
                    )
                    pr = PredictionResult(
                        process_status=ProcessStatus.ERROR,
                        error_reason="; ".join(validation.errors),
                        error_detail=iv_exc.error_detail,
                    )
                    vm = builder.build_single(
                        prediction_result=pr,
                        model_version_info=mv,
                        validation_status=vs,
                        trace_id=trace_id,
                    )
                    d = builder.flatten_for_display(vm)
                    _display_error_from_display(d)
                    return

                detailed = pipeline.predict_detailed(df, transaction_id=trace_id)
                vm = builder.build_single(
                    prediction_result=detailed,
                    model_version_info=mv,
                    validation_status=vs,
                    trace_id=trace_id,
                )
                d = builder.flatten_for_display(vm)

                if d.get("has_error"):
                    _display_error_from_display(d)
                    return

                st.subheader("Fraud Detector Class Labels")
                class_labels_df = pd.DataFrame({"Not Fraud": [0], "Fraud": [1]})
                class_labels_df.index = ["Class Labels"]
                st.dataframe(class_labels_df.T)

                _render_single(d)

                st.caption(f"Trace ID: {d.get('error_trace_id') or trace_id} · Model: v{d.get('model_version','')}")

            except AMLException as e:
                vm_err = UnifiedViewModel()
                vm_err.is_error = True
                vm_err.error_reason = e.message
                vm_err.process_status = "error"
                err = e.error_detail
                if err:
                    from aml_fraud_detector.constants import ERROR_SEVERITY
                    vm_err.error_code = err.error_code.value if hasattr(err.error_code, "value") else str(err.error_code)
                    vm_err.error_category = err.error_category.value if hasattr(err.error_category, "value") else str(err.error_category)
                    vm_err.error_category_display = ERROR_CATEGORY_DISPLAY.get(err.error_category, vm_err.error_category)
                    vm_err.error_message = err.message or ""
                    vm_err.error_field = err.field_name or ""
                    vm_err.error_value = str(err.field_value) if err.field_value is not None else ""
                    vm_err.error_trace_id = trace_id
                    vm_err.error_severity = ERROR_SEVERITY.get(err.error_code, "info")
                d_err = builder.flatten_for_display(vm_err)
                _display_error_from_display(d_err)
            except Exception as e:
                logging.error(f"Streamlit predict error: {e}", exc_info=True)
                err_resp = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
                vm_err = UnifiedViewModel()
                vm_err.is_error = True
                err_e = err_resp.error
                vm_err.error_reason = err_e.message if err_e else str(e)
                vm_err.process_status = "error"
                if err_e:
                    from aml_fraud_detector.constants import ERROR_SEVERITY
                    vm_err.error_code = err_e.error_code.value if hasattr(err_e.error_code, "value") else str(err_e.error_code)
                    vm_err.error_category = err_e.error_category.value if hasattr(err_e.error_category, "value") else str(err_e.error_category)
                    vm_err.error_category_display = ERROR_CATEGORY_DISPLAY.get(err_e.error_category, vm_err.error_category)
                    vm_err.error_message = err_e.message or ""
                    vm_err.error_field = err_e.field_name or ""
                    vm_err.error_value = str(err_e.field_value) if err_e.field_value is not None else ""
                    vm_err.error_trace_id = trace_id
                    vm_err.error_severity = ERROR_SEVERITY.get(err_e.error_code, "info")
                d_err = builder.flatten_for_display(vm_err)
                _display_error_from_display(d_err)
    else:
        st.info("📦 Upload a CSV file for batch prediction below")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded is not None and st.button("Predict Batch"):
            trace_id = _get_trace_id()
            try:
                batch_df = pd.read_csv(uploaded)
                tids = None
                if "transaction_id" in batch_df.columns:
                    tids = batch_df["transaction_id"].astype(str).tolist()
                    batch_df = batch_df.drop(columns=["transaction_id"])
                for c in ["from_bank", "to_bank"]:
                    if c in batch_df.columns:
                        batch_df[c] = batch_df[c].astype("object")

                br = pipeline.predict_batch(batch_df)
                if tids and len(tids) == len(br.predictions):
                    for i, p in enumerate(br.predictions):
                        if p.transaction_id is None or str(p.transaction_id).isdigit():
                            p.transaction_id = tids[i]

                vm = builder.build_batch(
                    batch_result=br,
                    model_version_info=mv,
                    validation_status=vs,
                    trace_id=trace_id,
                )
                d = builder.flatten_for_display(vm)

                if d.get("has_error"):
                    _display_error_from_display(d)
                    return

                bdf = builder.batch_to_dataframe(vm)
                _render_batch(d, bdf)

                st.caption(f"Trace ID: {d.get('error_trace_id') or trace_id} · Model: v{d.get('model_version','')}")

            except AMLException as e:
                vm_err = UnifiedViewModel()
                vm_err.batch_is_error = True
                vm_err.is_error = True
                vm_err.error_reason = e.message
                vm_err.batch_error_reason = e.message
                vm_err.process_status = "error"
                err = e.error_detail
                if err:
                    from aml_fraud_detector.constants import ERROR_SEVERITY
                    vm_err.error_code = err.error_code.value if hasattr(err.error_code, "value") else str(err.error_code)
                    vm_err.error_category = err.error_category.value if hasattr(err.error_category, "value") else str(err.error_category)
                    vm_err.error_category_display = ERROR_CATEGORY_DISPLAY.get(err.error_category, vm_err.error_category)
                    vm_err.error_message = err.message or ""
                    vm_err.error_field = err.field_name or ""
                    vm_err.error_value = str(err.field_value) if err.field_value is not None else ""
                    vm_err.error_trace_id = trace_id
                    vm_err.error_severity = ERROR_SEVERITY.get(err.error_code, "info")
                d_err = builder.flatten_for_display(vm_err)
                _display_error_from_display(d_err)
            except Exception as e:
                logging.error(f"Streamlit batch error: {e}", exc_info=True)
                err_resp = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
                vm_err = UnifiedViewModel()
                vm_err.batch_is_error = True
                vm_err.is_error = True
                err_e = err_resp.error
                vm_err.error_reason = err_e.message if err_e else str(e)
                vm_err.batch_error_reason = err_e.message if err_e else str(e)
                vm_err.process_status = "error"
                if err_e:
                    from aml_fraud_detector.constants import ERROR_SEVERITY
                    vm_err.error_code = err_e.error_code.value if hasattr(err_e.error_code, "value") else str(err_e.error_code)
                    vm_err.error_category = err_e.error_category.value if hasattr(err_e.error_category, "value") else str(err_e.error_category)
                    vm_err.error_category_display = ERROR_CATEGORY_DISPLAY.get(err_e.error_category, vm_err.error_category)
                    vm_err.error_message = err_e.message or ""
                    vm_err.error_field = err_e.field_name or ""
                    vm_err.error_value = str(err_e.field_value) if err_e.field_value is not None else ""
                    vm_err.error_trace_id = trace_id
                    vm_err.error_severity = ERROR_SEVERITY.get(err_e.error_code, "info")
                d_err = builder.flatten_for_display(vm_err)
                _display_error_from_display(d_err)

    st.write("---")
    st.markdown(
        """
        **Note:** This app is for demonstration purposes only. The predictions are based on a machine learning model.
        """
    )
    logging.info(f"Streamlit app execution completed")


if __name__ == "__main__":
    main()

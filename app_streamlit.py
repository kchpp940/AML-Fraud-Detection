import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from aml_fraud_detector.pipeline.prediction_pipeline import CustomData
from aml_fraud_detector.presentation.response_builder import (
    ResponseBuilder,
    BATCH_DISPLAY_COLUMNS,
)
from aml_fraud_detector.logger import logging


_builder = ResponseBuilder()


def _render_model(display: dict):
    """只读 display 字典，不做任何状态判断。"""
    st.subheader("Model Version Info")
    col1, col2 = st.columns(2)
    col1.metric("Model Version", display["model_version"])
    col1.metric("Model Name", display["model_name"])
    col2.metric("Trained At", display["training_time"])
    col2.metric(
        f"{display['selection_metric']} Score",
        f"{display['best_metric_value']:.4f}",
    )
    st.caption(f"Feature Contract: v{display['feature_contract_version']}")


def _render_validation(display: dict):
    """只读 display 字典，不做任何状态判断。"""
    st.subheader("Artifact Validation")
    # 成功/失败 banner：直接从 display 读 list 长度决定
    if display["validation_valid"]:
        st.success("All required artifacts present & digest verified.")
    for err in display["validation_errors"]:
        st.error(err)
    for warn in display["validation_warnings"]:
        st.warning(warn)
    details = display["validation_details"]
    with st.expander("Validation Details"):
        st.write({
            "artifacts_dir": details["artifacts_dir"],
            "manifest_present": details["manifest_present"],
            "manifest_artifacts": details["manifest_artifacts"],
            "digest_model_pkl": details.get("digest_model_pkl", "")[:16] + "..." if details.get("digest_model_pkl") else "",
            "digest_preprocessor_pkl": details.get("digest_preprocessor_pkl", "")[:16] + "..." if details.get("digest_preprocessor_pkl") else "",
            "numerical_features": details["numerical_features"],
            "categorical_features": details["categorical_features"],
        })


def _render_single(display: dict):
    """只读 display 中的 single_* 展示字段，不做任何判断。"""
    st.subheader("Prediction Result")

    # 错误 banner（空字符串 → 不显示）
    if display.get("single_error_banner"):
        st.error(display["single_error_banner"])
        return

    # 预测结果的展示风格（st.success / st.error）由 builder 预计算
    if display.get("single_prediction_streamlit_style") == "error":
        st.error(f"**{display['single_prediction_badge_label']}**")
    else:
        st.success(f"**{display['single_prediction_badge_label']}**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Fraud Probability", f"{display['fraud_probability']:.4f}")
    col2.metric("Legit Probability", f"{display['legit_probability']:.4f}")
    col3.markdown(
        f"<div style='font-size:1.1em;'>"
        f"<b>Risk:</b> "
        f"<span class='{display['single_risk_badge_class']}'>"
        f"{display['single_risk_badge_label']}</span></div>",
        unsafe_allow_html=True,
    )

    st.subheader("Prediction Probabilities")
    proba_df = pd.DataFrame({
        "Not Fraud": [display["legit_probability"]],
        "Fraud": [display["fraud_probability"]],
    })
    st.dataframe(proba_df)

    fig, ax = plt.subplots()
    ax.bar(proba_df.columns, proba_df.iloc[0], color=["green", "red"])
    ax.set_ylabel("Probability")
    ax.set_title("Fraud vs. Not Fraud Probability")
    st.pyplot(fig)

    st.subheader("Risk Explanation")
    st.info(display["risk_summary"])

    # Top Contributors（空字符串 → 不显示）
    contrib_text = display.get("single_risk_contributors_display") or ""
    if contrib_text:
        st.markdown("**Top Contributors**")
        st.code(contrib_text, language=None)

    st.caption(f"Transaction ID: {display['transaction_id']}")


def _render_batch(display: dict, batch_display_df: pd.DataFrame):
    """只读 display 中的 batch_* 展示字段 + 预先展开的批量明细表。"""
    st.subheader("Batch Prediction Summary")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", display["batch_summary_label_total"])
    col2.metric("Legit Count", display["batch_summary_label_legit"])
    col3.metric("Fraud Count", display["batch_summary_label_fraud"])
    col4.metric("Fraud Rate", display["batch_summary_label_fraud_rate"])

    # 批量级错误 banner
    if display.get("batch_error_banner"):
        st.error(f"Batch Error: {display['batch_error_banner']}")
        return

    if not batch_display_df.empty:
        st.subheader("Batch Detail Table")
        # 只挑出页面可见的列（避免展示内部 is_error_row 等字段）
        view_cols = [
            "row_index", "transaction_id",
            "prediction_badge_label", "fraud_probability_display",
            "legit_probability_display", "risk_level_label",
            "risk_summary_display", "risk_contributors_display",
            "status_label", "error_message_display",
        ]
        # 只保留 DataFrame 实际存在的列
        cols_to_show = [c for c in view_cols if c in batch_display_df.columns]
        df_view = batch_display_df[cols_to_show].rename(columns={
            "row_index": "#",
            "transaction_id": "Transaction ID",
            "prediction_badge_label": "Prediction",
            "fraud_probability_display": "Fraud Prob",
            "legit_probability_display": "Legit Prob",
            "risk_level_label": "Risk Level",
            "risk_summary_display": "Risk Summary",
            "risk_contributors_display": "Top Contributors",
            "status_label": "Status",
            "error_message_display": "Error",
        })
        st.dataframe(df_view, use_container_width=True)


def _collect_inputs() -> dict:
    st.sidebar.subheader("Transaction Details")
    from_bank = st.sidebar.number_input("From Bank", min_value=0, help="The bank ID from which the transaction originates.")
    account = st.sidebar.text_input("Account (Sender)", help="The account number of the sender.")
    to_bank = st.sidebar.number_input("To Bank", min_value=0, help="The bank ID to which the transaction is sent.")
    account_1 = st.sidebar.text_input("Account (Receiver)", help="The account number of the receiver.")
    amount_received = st.sidebar.number_input("Amount Received", min_value=0.0, help="The amount received in the transaction.")
    receiving_currency = st.sidebar.text_input("Receiving Currency", help="The currency in which the amount is received.")
    payment_currency = st.sidebar.text_input("Payment Currency", help="The currency used for the payment.")
    payment_format = st.sidebar.text_input("Payment Format", help="The format of the payment (e.g., wire transfer, check).")
    day = st.sidebar.text_input("Day", help="The day of the transaction.")
    data = CustomData(
        from_bank=from_bank, account=account, to_bank=to_bank, account_1=account_1,
        amount_received=amount_received, receiving_currency=receiving_currency,
        payment_currency=payment_currency, payment_format=payment_format, day=day,
    )
    return {
        "from_bank": data.from_bank, "account": data.account, "to_bank": data.to_bank,
        "account_1": data.account_1, "amount_received": data.amount_received,
        "receiving_currency": data.receiving_currency, "payment_currency": data.payment_currency,
        "payment_format": data.payment_format, "day": data.day,
    }, data.get_data_as_DataFrame()


def main():
    logging.info("Starting Streamlit App")

    st.title("Anti-Money Laundering (AML) Fraud Detection")
    st.markdown(
        """
        This app predicts whether a given transaction is **fraudulent** or **non-fraudulent**.
        Switch between the **Single** tab for one-shot input and the **Batch** tab for CSV uploads.
        """
    )
    st.write("---")

    # 先展示模型与校验信息（无论单条/批量）
    vm_val = _builder.build_validation_only()
    display_val = ResponseBuilder.flatten_for_display(vm_val)
    _render_model(display_val)
    _render_validation(display_val)
    st.write("---")

    tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])

    # ========================================================================
    # 单条预测
    # ========================================================================
    with tab_single:
        st.sidebar.header("Specify Input Features")
        input_dict, input_df = _collect_inputs()

        st.subheader("Specified Input Parameters")
        st.dataframe(input_df)

        if st.button("Predict"):
            vm = _builder.build_single(input_dict)
            display = ResponseBuilder.flatten_for_display(vm)
            _render_single(display)

    # ========================================================================
    # 批量预测
    # ========================================================================
    with tab_batch:
        st.info(
            "CSV needs columns: from_bank, account, to_bank, account_1, "
            "amount_received, receiving_currency, payment_currency, payment_format, day. "
            "Optional: transaction_id."
        )
        uploaded = st.file_uploader("Upload batch CSV (UTF-8, max 16MB)", type=["csv", "txt"])

        if uploaded is not None:
            try:
                batch_in_df = pd.read_csv(uploaded)
                st.caption(f"Input rows: {len(batch_in_df)}")
                with st.expander("Preview Input"):
                    st.dataframe(batch_in_df.head(20))
                if st.button("Run Batch Prediction", key="batch_run"):
                    vm = _builder.build_batch(batch_in_df)
                    display = ResponseBuilder.flatten_for_display(vm)
                    batch_df = ResponseBuilder.batch_to_display_dataframe(vm)
                    _render_batch(display, batch_df)
            except Exception as exc:
                st.error(f"Failed to read CSV: {exc}")

    st.write("---")
    st.markdown(
        """
        **Note:** This app is for demonstration purposes only. The predictions are based on a machine learning model.
        """
    )
    logging.info("Streamlit app execution completed")


if __name__ == "__main__":
    main()

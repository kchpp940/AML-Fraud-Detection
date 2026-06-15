import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from aml_fraud_detector.pipeline.prediction_pipeline import CustomData
from aml_fraud_detector.presentation.response_builder import ResponseBuilder
from aml_fraud_detector.logger import logging


_builder = ResponseBuilder()


def _render_model(display: dict):
    st.subheader("Model Version Info")
    col1, col2 = st.columns(2)
    col1.metric("Model Version", display["model_version"])
    col1.metric("Model Name", display["model_name"])
    col2.metric("Trained At", display["training_time"])
    col2.metric(f"{display['selection_metric']} Score", f"{display['best_metric_value']:.4f}")
    st.caption(f"Feature Contract: v{display['feature_contract_version']}")


def _render_validation(display: dict):
    st.subheader("Artifact Validation")
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
    if display.get("is_error"):
        st.subheader("Prediction Error")
        st.error(display["error_reason"])
        return

    st.subheader("Prediction Result")
    if display["prediction_code"] == 1:
        st.error(f"**Fraudulent Transaction** — {display['prediction_label']}")
    else:
        st.success(f"**Non-Fraudulent Transaction** — {display['prediction_label']}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Fraud Probability", f"{display['fraud_probability']:.4f}")
    col2.metric("Legit Probability", f"{display['legit_probability']:.4f}")
    col3.metric("Risk Level", display["risk_level"])

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

    contribs = display.get("risk_contributors") or []
    if contribs:
        st.markdown("**Top Contributors**")
        cdf = pd.DataFrame(contribs)
        st.dataframe(cdf.style.format({"contribution": "{:.5f}"}))

    st.caption(f"Transaction ID: {display['transaction_id']}")


def _render_batch(display: dict, vm):
    st.subheader("Batch Prediction Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", display["batch_total"])
    col2.metric("Fraud Count", display["batch_fraud"])
    col3.metric("Legit Count", display["batch_legit"])
    col4.metric("Fraud Rate", f"{display['batch_fraud_rate']:.2%}")

    if display.get("batch_is_error"):
        st.error(f"Batch Error: {display['batch_error_reason']}")
        return

    flat_df = ResponseBuilder.batch_to_dataframe(vm)
    if not flat_df.empty:
        st.subheader("Batch Detail Table")
        st.dataframe(flat_df)


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
        Please provide the required input features in the sidebar and click the **Predict** button.
        """
    )
    st.write("---")

    st.sidebar.header("Specify Input Features")
    input_dict, input_df = _collect_inputs()

    st.header("Specified Input Parameters")
    st.dataframe(input_df)
    st.write("---")

    # 先展示模型与校验信息（无需点击预测也可见）
    vm_val = _builder.build_validation_only()
    display_val = ResponseBuilder.flatten_for_display(vm_val)
    _render_model(display_val)
    _render_validation(display_val)
    st.write("---")

    st.header("Prediction Results")
    if st.button("Predict"):
        vm = _builder.build_single(input_dict)
        display = ResponseBuilder.flatten_for_display(vm)
        _render_single(display)

    st.write("---")
    st.markdown(
        """
        **Note:** This app is for demonstration purposes only. The predictions are based on a machine learning model.
        """
    )
    logging.info("Streamlit app execution completed")


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.presentation.response_builder import ResponseBuilder
from aml_fraud_detector.logger import logging


_pipeline = PredictionPipeline()
_builder = ResponseBuilder(_pipeline)


def _render_model_info(display: dict):
    st.subheader("Model Version Info")
    col1, col2 = st.columns(2)
    col1.metric("Model Version", display["model_version_number"])
    col1.metric("Model Name", display["model_name"])
    col2.metric("Trained At", display["training_time"])
    col2.metric(f"{display['selection_metric']} Score", f"{display['best_metric_value']:.4f}")


def _render_validation_alerts(display: dict):
    if display.get("has_alerts"):
        st.subheader("Validation Alerts")
        for err in display.get("validation_errors", []):
            st.error(f"❌ {err}")
        for warn in display.get("validation_warnings", []):
            st.warning(f"⚠️ {warn}")


def _render_single_prediction(display: dict):
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
    risk = display["risk_level"]
    col3.metric("Risk Level", risk)

    st.subheader("Prediction Probabilities")
    proba_df = pd.DataFrame(
        {"Not Fraud": [display["legit_probability"]], "Fraud": [display["fraud_probability"]]}
    )
    st.dataframe(proba_df)

    fig, ax = plt.subplots()
    ax.bar(proba_df.columns, proba_df.iloc[0], color=["green", "red"])
    ax.set_ylabel("Probability")
    ax.set_title("Fraud vs. Not Fraud Probability")
    st.pyplot(fig)

    if display.get("transaction_id"):
        st.caption(f"Transaction ID: {display['transaction_id']}")


def _render_batch_prediction(display: dict, response):
    batch = response.batch_prediction
    if batch is None:
        return

    st.subheader("Batch Prediction Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total", display["batch_total"])
    col2.metric("Fraud Count", display["batch_fraud_count"])
    col3.metric("Fraud Rate", f"{display['batch_fraud_rate']:.2%}")

    if display.get("batch_is_error"):
        st.error(f"Batch Error: {display['batch_error_reason']}")
        return

    flat_df = _builder.to_flat_dataframe(response)
    if not flat_df.empty:
        st.subheader("Batch Detail Table")
        st.dataframe(flat_df)


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
        day = st.sidebar.text_input("Day", help="The day of the transaction.")

        data = CustomData(
            from_bank=from_bank,
            account=account,
            to_bank=to_bank,
            account_1=account_1,
            amount_received=amount_received,
            receiving_currency=receiving_currency,
            payment_currency=payment_currency,
            payment_format=payment_format,
            day=day,
        )
        return data

    custom_data = user_input_features()
    df = custom_data.get_data_as_DataFrame()

    st.header("Specified Input Parameters")
    st.dataframe(df)
    st.write("---")

    st.header("Prediction Results")

    if st.button("Predict"):
        response = _builder.build_single_response(custom_data.to_dict())
        display = ResponseBuilder.extract_display_fields(response)

        _render_model_info(display)
        _render_validation_alerts(display)
        _render_single_prediction(display)

    st.write("---")
    st.markdown(
        """
        **Note:** This app is for demonstration purposes only. The predictions are based on a machine learning model.
        """
    )
    logging.info("Streamlit app execution completed")


if __name__ == "__main__":
    main()

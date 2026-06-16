import streamlit as st
import dill
import pandas as pd
import matplotlib.pyplot as plt
from aml_fraud_detector.bootstrap import ApplicationContainer
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging


def _init_container() -> ApplicationContainer:
    if "app_container" not in st.session_state:
        ApplicationContainer.reset()
        st.session_state.app_container = ApplicationContainer()
    return st.session_state.app_container


def _render_sidebar_health(container: ApplicationContainer) -> None:
    health = container.health
    model_version = container.get_model_version()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Service Status")

    overall_status = health.overall.value
    if overall_status == "healthy":
        st.sidebar.success(f"**Status:** {overall_status.upper()}")
    elif overall_status == "degraded":
        st.sidebar.warning(f"**Status:** {overall_status.upper()}")
    else:
        st.sidebar.error(f"**Status:** {overall_status.upper()}")

    with st.sidebar.expander("Model Information", expanded=True):
        if model_version.model_version:
            st.write(f"**Version:** v{model_version.model_version}")
        if model_version.model_name:
            st.write(f"**Model:** {model_version.model_name}")
        if model_version.selection_metric:
            st.write(f"**Metric:** {model_version.selection_metric}")
        if model_version.best_metric_value:
            st.write(f"**Score:** {model_version.best_metric_value:.4f}")
        if model_version.training_time:
            st.write(f"**Trained:** {model_version.training_time}")
        if model_version.feature_schema_version:
            st.write(f"**Schema:** v{model_version.feature_schema_version}")

    with st.sidebar.expander("Component Details", expanded=False):
        for comp in health.components:
            if comp.status == "ok":
                status_icon = "✅"
            elif comp.status == "warning":
                status_icon = "⚠️"
            elif comp.status == "error":
                status_icon = "❌"
            else:
                status_icon = "⏭️"
            st.write(f"{status_icon} **{comp.name}**: {comp.status}")
            if comp.duration_ms > 0:
                st.caption(f"  Duration: {comp.duration_ms:.2f}ms")
            if comp.message:
                st.caption(f"  {comp.message}")

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Started: {health.started_at}")
    if health.checked_at:
        st.sidebar.caption(f"Last check: {health.checked_at}")


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
        day=day
    )
    features_df = data.get_data_as_DataFrame()
    return features_df


def main():
    logging.info("Starting Streamlit App")

    container = _init_container()
    health = container.check_health()

    st.title("Anti-Money Laundering (AML) Fraud Detection")
    st.markdown(
        """
        This app predicts whether a given transaction is **fraudulent** or **non-fraudulent**.
        Please provide the required input features in the sidebar and click the **Predict** button.
        """
    )
    st.write("---")

    if not health.is_healthy():
        st.error(
            f"⚠️ Service is not healthy (status: {health.overall.value}). "
            "Prediction may not work correctly. Please check the sidebar for details."
        )
        st.write("---")

    st.sidebar.header("Specify Input Features")

    _render_sidebar_health(container)

    df = user_input_features()

    st.header("Specified Input Parameters")
    st.dataframe(df)
    st.write("---")

    st.header("Prediction Results")

    if container.prediction_pipeline is not None:
        predict_pipeline = container.prediction_pipeline
    else:
        predict_pipeline = PredictionPipeline()
        st.warning("Using fallback PredictionPipeline. Service may be degraded.")

    if st.button("Predict"):
        if not health.is_healthy():
            st.warning("Service is not fully healthy. Prediction results may be unreliable.")

        prediction = predict_pipeline.predict(df)
        prediction_proba = predict_pipeline.predict_proba(df)

        st.subheader("Fraud Detector Class Labels")
        class_labels_df = pd.DataFrame({"Not Fraud": [0], "Fraud": [1]})
        class_labels_df.index = ["Class Labels"]
        st.dataframe(class_labels_df.T)

        st.subheader("Prediction of the Given Transaction")
        if prediction.prediction == 1:
            st.error("**Fraudulent Transaction**")
        else:
            st.success("**Non-Fraudulent Transaction**")

        st.subheader("Prediction Probabilities")
        proba_df = pd.DataFrame(prediction_proba, columns=["Not Fraud", "Fraud"])
        st.dataframe(proba_df)

        st.subheader("Prediction Probability Distribution")
        fig, ax = plt.subplots()
        ax.bar(proba_df.columns, proba_df.iloc[0], color=["green", "red"])
        ax.set_ylabel("Probability")
        ax.set_title("Fraud vs. Not Fraud Probability")
        st.pyplot(fig)

    st.write("---")
    st.markdown(
        """
        **Note:** This app is for demonstration purposes only. The predictions are based on a machine learning model.
        """
    )
    logging.info("Streamlit app execution completed")


if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging


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
            from_bank=int(from_bank),
            account=account or "",
            to_bank=int(to_bank),
            account_1=account_1 or "",
            amount_received=float(amount_received),
            receiving_currency=receiving_currency or "",
            payment_currency=payment_currency or "",
            payment_format=payment_format or "",
            day=day or "",
        )
        features_df = data.get_data_as_DataFrame()
        return features_df, data

    df, transaction_data = user_input_features()

    st.header("Specified Input Parameters")
    st.dataframe(df)
    st.write("---")

    st.header("Prediction Results")
    predict_pipeline = PredictionPipeline()

    if st.button("Predict"):
        prediction = predict_pipeline.predict(df)
        prediction_proba = predict_pipeline.predict_proba(df)

        st.subheader("Fraud Detector Class Labels")
        class_labels_df = pd.DataFrame({"Not Fraud": [0], "Fraud": [1]})
        class_labels_df.index = ["Class Labels"]
        st.dataframe(class_labels_df.T)

        st.subheader("Prediction of the Given Transaction")
        if prediction[0] == 1:
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

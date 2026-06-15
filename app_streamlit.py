import io
import streamlit as st
import dill
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline, REQUIRED_FIELDS
from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
import pandas as pd
import matplotlib.pyplot as plt


def main():
    logging.info(f"Starting Streamlit App")

    st.set_page_config(
        page_title="AML Fraud Detection",
        page_icon="🔍",
        layout="wide"
    )

    st.title("Anti-Money Laundering (AML) Fraud Detection")
    st.markdown(
        """
        This app predicts whether a given transaction is **fraudulent** or **non-fraudulent**.
        Choose between **Single Transaction** or **Batch Prediction** mode below.
        """
    )
    st.write("---")

    predict_pipeline = PredictionPipeline()

    mode = st.radio(
        "Select Prediction Mode",
        ["Single Transaction", "Batch Prediction (CSV Upload)"],
        horizontal=True
    )

    st.write("---")

    if mode == "Single Transaction":
        single_transaction_mode(predict_pipeline)
    else:
        batch_prediction_mode(predict_pipeline)

    st.write("---")
    st.markdown(
        """
        **Note:** This app is for demonstration purposes only. The predictions are based on a machine learning model.
        """
    )
    logging.info(f"Streamlit app execution completed")


def single_transaction_mode(predict_pipeline):
    st.header("Single Transaction Prediction")

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
        day = st.sidebar.text_input("Day", help="The day of the transaction (e.g., Monday).")

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

    df = user_input_features()

    st.subheader("Specified Input Parameters")
    st.dataframe(df, use_container_width=True)
    st.write("---")

    st.subheader("Prediction Results")

    if st.button("Predict", type="primary"):
        with st.spinner("Analyzing transaction..."):
            prediction = predict_pipeline.predict(df)
            prediction_proba = predict_pipeline.predict_proba(df)

        st.subheader("Fraud Detector Class Labels")
        class_labels_df = pd.DataFrame({"Not Fraud": [0], "Fraud": [1]})
        class_labels_df.index = ["Class Labels"]
        st.dataframe(class_labels_df.T)

        st.subheader("Prediction of the Given Transaction")
        if prediction[0] == 1:
            st.error("**🚨 Fraudulent Transaction**")
        else:
            st.success("**✅ Non-Fraudulent Transaction**")

        st.subheader("Prediction Probabilities")
        proba_df = pd.DataFrame(prediction_proba, columns=["Not Fraud", "Fraud"])
        st.dataframe(proba_df)

        st.subheader("Prediction Probability Distribution")
        fig, ax = plt.subplots()
        ax.bar(proba_df.columns, proba_df.iloc[0], color=["green", "red"])
        ax.set_ylabel("Probability")
        ax.set_title("Fraud vs. Not Fraud Probability")
        st.pyplot(fig)


def batch_prediction_mode(predict_pipeline):
    st.header("Batch Transaction Prediction")

    st.info(
        """
        **📋 Required CSV Columns:**
        """
    )

    fields_df = pd.DataFrame([
        {
            "Column Name": field,
            "Type": dtype.__name__,
            "Description": {
                "from_bank": "汇出银行ID (整数)",
                "account": "汇款人账号 (字符串)",
                "to_bank": "汇入银行ID (整数)",
                "account_1": "收款人账号 (字符串)",
                "amount_received": "到账金额 (浮点数)",
                "receiving_currency": "收款币种 (字符串)",
                "payment_currency": "付款币种 (字符串)",
                "payment_format": "付款方式 (字符串)",
                "day": "交易星期 (字符串, e.g., Monday)"
            }.get(field, "")
        }
        for field, dtype in REQUIRED_FIELDS.items()
    ])
    st.dataframe(fields_df, use_container_width=True, hide_index=True)

    st.write("---")

    st.subheader("Upload CSV File")
    uploaded_file = st.file_uploader(
        "Choose a CSV file containing transactions",
        type=["csv"],
        help="Upload a CSV file with the required columns listed above."
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Successfully loaded {len(input_df)} transactions")

            with st.expander("Preview Uploaded Data", expanded=False):
                st.dataframe(input_df.head(10), use_container_width=True)
                st.caption(f"Showing first 10 of {len(input_df)} rows")

            st.write("---")

            if st.button("🔍 Run Batch Prediction", type="primary"):
                with st.spinner(f"Processing {len(input_df)} transactions..."):
                    result_df = predict_pipeline.predict_batch(input_df)

                display_batch_results(result_df, input_df)

        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            logging.error(f"Error reading CSV file: {str(e)}")


def display_batch_results(result_df, input_df):
    st.write("---")
    st.subheader("📊 Prediction Results Summary")

    total_count = len(result_df)
    success_count = result_df["prediction_label"].notna().sum()
    fail_count = result_df["prediction_label"].isna().sum()
    fraud_count = (result_df["prediction_label"] == 1).sum()
    not_fraud_count = (result_df["prediction_label"] == 0).sum()
    warning_count = result_df["error_reason"].notna().sum() - fail_count

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Transactions", total_count)
    col2.metric("Successfully Processed", f"{success_count}",
                delta_color="normal" if success_count == total_count else "off")
    col3.metric("With Warnings", f"{warning_count}", delta_color="off")
    col4.metric("Failed", f"{fail_count}", delta_color="inverse")
    col5.metric("🚨 Fraudulent", f"{fraud_count}", delta_color="inverse")
    col6.metric("✅ Legitimate", f"{not_fraud_count}")

    st.write("---")

    tab1, tab2, tab3 = st.tabs(["All Results", "Successful Predictions", "Failed Transactions"])

    with tab1:
        st.dataframe(
            result_df,
            use_container_width=True,
            column_config={
                "prediction_label": st.column_config.NumberColumn(
                    "Prediction",
                    format="%d",
                    help="0 = Not Fraud, 1 = Fraud"
                ),
                "fraud_probability": st.column_config.ProgressColumn(
                    "Fraud Probability",
                    format="%.2f",
                    min_value=0,
                    max_value=1
                ),
                "error_reason": st.column_config.TextColumn(
                    "Error Reason",
                    width="large"
                )
            }
        )

        successful_df = result_df[result_df["error_reason"].isna()]
        if len(successful_df) > 0:
            st.subheader("Fraud Probability Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(successful_df["fraud_probability"], bins=20, edgecolor="black", alpha=0.7)
            ax.set_xlabel("Fraud Probability")
            ax.set_ylabel("Number of Transactions")
            ax.set_title("Distribution of Fraud Probabilities")
            ax.axvline(x=0.5, color="red", linestyle="--", label="Threshold (0.5)")
            ax.legend()
            st.pyplot(fig)

    with tab2:
        successful_df = result_df[result_df["error_reason"].isna()].copy()
        successful_df = successful_df.drop(columns=["error_reason"])
        if len(successful_df) > 0:
            st.dataframe(
                successful_df,
                use_container_width=True,
                column_config={
                    "prediction_label": st.column_config.NumberColumn(
                        "Prediction",
                        format="%d"
                    ),
                    "fraud_probability": st.column_config.ProgressColumn(
                        "Fraud Probability",
                        format="%.2f",
                        min_value=0,
                        max_value=1
                    )
                }
            )

            fraud_df = successful_df[successful_df["prediction_label"] == 1]
            if len(fraud_df) > 0:
                st.subheader("🚨 High Risk Transactions")
                st.warning(f"Found {len(fraud_df)} potentially fraudulent transactions")
                st.dataframe(fraud_df.sort_values("fraud_probability", ascending=False),
                           use_container_width=True)
        else:
            st.info("No successful predictions.")

    with tab3:
        failed_df = result_df[result_df["error_reason"].notna()].copy()
        if len(failed_df) > 0:
            st.dataframe(
                failed_df,
                use_container_width=True,
                column_config={
                    "error_reason": st.column_config.TextColumn(
                        "Error Reason",
                        width="large"
                    )
                }
            )

            st.subheader("Common Errors")
            error_counts = failed_df["error_reason"].value_counts().head(10)
            error_df = pd.DataFrame({
                "Error Type": error_counts.index,
                "Count": error_counts.values
            })
            st.dataframe(error_df, use_container_width=True, hide_index=True)
        else:
            st.success("All transactions processed successfully!")

    st.write("---")
    st.subheader("📥 Download Results")

    csv = result_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="⬇️ Download Full Results as CSV",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv",
        type="secondary"
    )


if __name__ == "__main__":
    main()

import streamlit as st
import dill
from aml_fraud_detector.pipeline.prediction_pipeline import (
    CustomData, PredictionPipeline, InputValidationError
)
from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
import pandas as pd
import matplotlib.pyplot as plt


WEEKDAY_OPTIONS = [
    ("Monday (周一)", "Monday"),
    ("Tuesday (周二)", "Tuesday"),
    ("Wednesday (周三)", "Wednesday"),
    ("Thursday (周四)", "Thursday"),
    ("Friday (周五)", "Friday"),
    ("Saturday (周六)", "Saturday"),
    ("Sunday (周日)", "Sunday"),
]


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

        col1, col2 = st.sidebar.columns(2)
        with col1:
            from_bank = st.number_input(
                "From Bank (发起银行)",
                min_value=0,
                step=1,
                help="The bank ID from which the transaction originates.",
                value=70,
            )
        with col2:
            to_bank = st.number_input(
                "To Bank (接收银行)",
                min_value=0,
                step=1,
                help="The bank ID to which the transaction is sent.",
                value=1502,
            )

        account = st.sidebar.text_input(
            "Account (Sender) 发起账户",
            value="1004286A8",
            help="The account number / code of the sender.",
        )
        account_1 = st.sidebar.text_input(
            "Account (Receiver) 接收账户",
            value="812191060",
            help="The account number / code of the receiver.",
        )

        amount_received = st.sidebar.text_input(
            "Amount Received (交易金额)",
            value="46480.59",
            help="The amount received in the transaction. Supports comma separators, e.g. 1,234,567.89",
        )

        col3, col4 = st.sidebar.columns(2)
        with col3:
            receiving_currency = st.sidebar.text_input(
                "Receiving Currency",
                value="Euro",
                help="The currency in which the amount is received.",
            )
        with col4:
            payment_currency = st.sidebar.text_input(
                "Payment Currency",
                value="Euro",
                help="The currency used for the payment.",
            )

        payment_format = st.sidebar.text_input(
            "Payment Format (支付方式)",
            value="Cheque",
            help="The format of the payment (e.g., Cheque, Credit Card, Wire, ACH).",
        )

        st.sidebar.subheader("Date / Weekday")
        day_mode = st.sidebar.radio(
            "选择日期或星期",
            ["日期 (Date)", "星期 (Weekday)", "自由输入 (Manual)"],
            horizontal=True,
        )
        day = None
        if day_mode == "日期 (Date)":
            d = st.sidebar.date_input("Transaction Date")
            day = d.strftime("%Y-%m-%d") if d else None
        elif day_mode == "星期 (Weekday)":
            label = st.sidebar.selectbox(
                "Weekday",
                options=[o[0] for o in WEEKDAY_OPTIONS],
                index=0,
            )
            for lbl, val in WEEKDAY_OPTIONS:
                if lbl == label:
                    day = val
                    break
        else:
            day = st.sidebar.text_input(
                "Day (自由输入)",
                value="Wednesday",
                help="支持格式: 日期(YYYY-MM-DD)、星期名(Monday/周一)、星期数字(0-6: 周一-周日)",
            )

        return {
            "from_bank": from_bank,
            "account": account,
            "to_bank": to_bank,
            "account_1": account_1,
            "amount_received": amount_received,
            "receiving_currency": receiving_currency,
            "payment_currency": payment_currency,
            "payment_format": payment_format,
            "day": day,
        }

    raw_inputs = user_input_features()

    st.header("Specified Input Parameters")
    input_display = pd.DataFrame(
        [[raw_inputs[k] for k in raw_inputs]],
        columns=list(raw_inputs.keys()),
    )
    st.dataframe(input_display, use_container_width=True)
    st.write("---")

    st.header("Prediction Results")
    predict_pipeline = PredictionPipeline()

    if st.button("Predict"):
        try:
            with st.spinner("Validating inputs and running prediction..."):
                data = CustomData(**raw_inputs)
                features_df = data.get_data_as_DataFrame()

                with st.expander("查看规范化后的数据（已送入模型）"):
                    st.dataframe(features_df.T, use_container_width=True)

                prediction = predict_pipeline.predict(features_df)
                prediction_proba = predict_pipeline.predict_proba(features_df)

            st.subheader("Fraud Detector Class Labels")
            class_labels_df = pd.DataFrame({"Not Fraud": [0], "Fraud": [1]})
            class_labels_df.index = ["Class Labels"]
            st.dataframe(class_labels_df.T)

            st.subheader("Prediction of the Given Transaction")
            if prediction[0] == 1:
                st.error("🚨 **Fraudulent Transaction（可疑欺诈交易）**")
            else:
                st.success("✅ **Non-Fraudulent Transaction（正常交易）**")

            st.subheader("Prediction Probabilities")
            proba_df = pd.DataFrame(
                prediction_proba, columns=["Not Fraud", "Fraud"]
            )
            st.dataframe(proba_df, use_container_width=True)

            st.subheader("Prediction Probability Distribution")
            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.bar(
                proba_df.columns, proba_df.iloc[0],
                color=["#27ae60", "#e74c3c"], edgecolor="#2c3e50",
            )
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            ax.set_title("Fraud vs. Not Fraud Probability")
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=9,
                )
            st.pyplot(fig)

        except InputValidationError as e:
            logging.warning(f"Input validation failed: {e}")
            errors = str(e).split("；")
            st.error("⚠️ **输入校验失败，请检查以下字段：**")
            for idx, msg in enumerate(errors, 1):
                st.markdown(f"{idx}. {msg}")

        except CustomerException as e:
            logging.error(f"Prediction pipeline error: {e}")
            st.error(
                f"❌ **预测服务异常，请稍后重试或联系管理员。**\n\n"
                f"详情: `{e.error_message}`"
            )

        except Exception as e:
            logging.error(f"Unexpected error: {e}", exc_info=True)
            st.error("❌ **服务器内部错误，请稍后重试。**")

    st.write("---")
    st.markdown(
        """
        **Note:** This app is for demonstration purposes only.
        The predictions are based on a machine learning model.
        """
    )
    logging.info("Streamlit app execution completed")


if __name__ == "__main__":
    main()

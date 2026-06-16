import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.exception import (
    AMLException,
    create_error_from_exception,
    create_error_response,
)
from aml_fraud_detector.constants import ErrorCode, ErrorCategory
from aml_fraud_detector.entity import (
    PredictionResult,
    ValidationStatus,
    ProcessStatus,
    ErrorDetail,
)
from aml_fraud_detector.logger import logging


def _display_error_from_detail(error_detail: ErrorDetail):
    category_icons = {
        ErrorCategory.DATA_QUALITY: "📊",
        ErrorCategory.INPUT_VALIDATION: "✏️",
        ErrorCategory.FEATURE_ALIGNMENT: "🔄",
        ErrorCategory.MODEL_LOADING: "📦",
        ErrorCategory.METADATA_VALIDATION: "📋",
        ErrorCategory.PREDICTION_ERROR: "⚠️",
        ErrorCategory.EXPLANATION_ERROR: "💡",
        ErrorCategory.CONFIG_ERROR: "⚙️",
        ErrorCategory.PIPELINE_ERROR: "🔧",
        ErrorCategory.INTERNAL_ERROR: "❌",
    }
    icon = category_icons.get(error_detail.error_category, "❌")

    suggestions = {
        ErrorCategory.DATA_QUALITY: "请检查数据源文件是否存在且格式正确",
        ErrorCategory.INPUT_VALIDATION: "请检查输入字段是否符合要求",
        ErrorCategory.FEATURE_ALIGNMENT: "请确认输入数据特征与训练时一致",
        ErrorCategory.MODEL_LOADING: "请检查模型文件是否存在或稍后重试",
        ErrorCategory.METADATA_VALIDATION: "请检查模型产物完整性，可能需要重新训练",
        ErrorCategory.PREDICTION_ERROR: "预测执行失败，请稍后重试",
        ErrorCategory.EXPLANATION_ERROR: "解释生成失败，可尝试其他模型",
        ErrorCategory.CONFIG_ERROR: "请检查配置文件是否正确",
        ErrorCategory.PIPELINE_ERROR: "管道执行失败，请查看日志详情",
        ErrorCategory.INTERNAL_ERROR: "系统内部错误，请联系技术支持",
    }
    suggestion = suggestions.get(error_detail.error_category, "请稍后重试")

    st.error(f"{icon} **预测失败** [{error_detail.error_code.value}]")
    st.error(f"**类别:** {error_detail.error_category.value} | **消息:** {error_detail.message}")
    st.info(f"💡 {suggestion}")

    with st.expander("查看详细错误信息"):
        st.json(error_detail.to_dict())


def _display_error_from_exception(e: Exception):
    if isinstance(e, AMLException):
        _display_error_from_detail(e.error_detail)
    else:
        error_resp = create_error_from_exception(e)
        if error_resp.error:
            _display_error_from_detail(error_resp.error)
        else:
            st.error(f"发生未知错误: {str(e)}")


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
            day=day
        )

        validation = data.validate()
        if not validation.is_valid:
            st.sidebar.warning(f"输入校验警告: {'; '.join(validation.errors)}")

        try:
            features_df = data.get_data_as_DataFrame()
            return features_df, validation
        except AMLException as e:
            _display_error_from_detail(e.error_detail)
            return None, validation
        except Exception as e:
            _display_error_from_exception(e)
            return None, validation

    df, validation = user_input_features()

    if df is not None:
        st.header("Specified Input Parameters")
        st.dataframe(df)
    else:
        st.header("Specified Input Parameters")
        st.warning("无法生成输入数据，请检查侧边栏输入")
    st.write("---")

    st.header("Prediction Results")
    predict_pipeline = PredictionPipeline()

    if st.button("Predict"):
        if df is None:
            error_resp = create_error_response(
                ErrorCode.INPUT_INVALID_FORMAT,
                field="input_data",
                value="无法生成有效的输入数据",
            )
            _display_error_from_detail(error_resp.error)
        elif not validation.is_valid:
            error_resp = create_error_response(
                ErrorCode.INPUT_INVALID_FORMAT,
                field="multiple",
                value="; ".join(validation.errors),
            )
            _display_error_from_detail(error_resp.error)
        else:
            result = predict_pipeline.predict(df)

            if not result.is_success():
                _display_error_from_detail(result.error_detail)
            else:
                st.subheader("Fraud Detector Class Labels")
                class_labels_df = pd.DataFrame({"Not Fraud": [0], "Fraud": [1]})
                class_labels_df.index = ["Class Labels"]
                st.dataframe(class_labels_df.T)

                st.subheader("Prediction of the Given Transaction")
                if result.prediction == 1:
                    st.error("**Fraudulent Transaction**")
                else:
                    st.success("**Non-Fraudulent Transaction**")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fraud Probability", f"{result.fraud_probability:.4f}")
                with col2:
                    st.metric("Legit Probability", f"{result.legit_probability:.4f}")
                with col3:
                    risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}
                    st.metric("Risk Level", f"{risk_emoji.get(result.risk_level.value, '⚪')} {result.risk_level.value}")

                if result.model_version > 0:
                    st.caption(f"Model Version: {result.model_version}")

                st.subheader("Prediction Probability Distribution")
                fig, ax = plt.subplots()
                probs = [result.legit_probability, result.fraud_probability]
                colors = ["green", "red"]
                labels = ["Not Fraud", "Fraud"]
                ax.bar(labels, probs, color=colors)
                ax.set_ylabel("Probability")
                ax.set_title("Fraud vs. Not Fraud Probability")
                for i, v in enumerate(probs):
                    ax.text(i, v + 0.01, f"{v:.4f}", ha="center")
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

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
from aml_fraud_detector.logger import logging
import pandas as pd
import matplotlib.pyplot as plt


def _get_trace_id():
    return str(uuid.uuid4())


def _display_error(error_resp):
    if isinstance(error_resp, ErrorDetail):
        error_resp = UnifiedErrorResponse(
            success=False,
            status="error",
            error=error_resp,
            http_status=HTTP_STATUS_CODES.get(error_resp.error_category, 500),
            trace_id=_get_trace_id(),
        )
    display = error_resp.to_user_display()
    cat_label = ERROR_CATEGORY_DISPLAY.get(display["error_category"], display["error_category"])

    icon = "⚠️"
    if display.get("severity") == "critical":
        icon = "🚨"
    elif display.get("severity") == "info":
        icon = "ℹ️"

    st.markdown(
        f"""
        <div style="
            padding: 16px;
            border-radius: 8px;
            background: #fff5f5;
            border-left: 6px solid #e53e3e;
            margin-bottom: 16px;
        ">
            <div style="font-size: 16px; font-weight: 600; color: #c53030; margin-bottom: 8px;">
                {icon} {cat_label} · <code style="background:#fed7d7;padding:2px 6px;border-radius:3px;">{display["error_code"]}</code>
            </div>
            <div style="font-size: 14px; color: #742a2a; margin-bottom: 6px; line-height: 1.5;">
                {display["message"]}
            </div>
            {"<div style=\"font-size: 12px; color: #9b2c2c; margin-bottom: 4px;\"><strong>字段：</strong>" + display["field"] + "</div>" if display.get("field") else ""}
            {"<div style=\"font-size: 12px; color: #9b2c2c; margin-bottom: 6px;\"><strong>建议：</strong>" + display["suggestion"] + "</div>" if display.get("suggestion") else ""}
            <div style="font-size: 11px; color: #b7791f; font-family: monospace; margin-top: 8px;">
                Trace ID: {display["trace_id"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _load_pipeline():
    return PredictionPipeline()


def main():
    logging.info(f"Starting Streamlit App")

    # App Title and Description
    st.title("Anti-Money Laundering (AML) Fraud Detection")
    st.markdown(
        """
        This app predicts whether a given transaction is **fraudulent** or **non-fraudulent**.
        Please provide the required input features in the sidebar and click the **Predict** button.
        """
    )
    st.write("---")

    # Sidebar for Input Features
    st.sidebar.header("Specify Input Features")

    try:
        pipeline = _load_pipeline()
        mv = pipeline.get_model_version_info()
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Model Info")
        st.sidebar.markdown(f"- **Name**: {mv.model_name}")
        st.sidebar.markdown(f"- **Version**: v{mv.model_version}")
        st.sidebar.markdown(f"- **Schema**: v{mv.feature_schema_version}")
        st.sidebar.markdown("✅ Loaded")
    except AMLException as e:
        st.sidebar.error(f"Model load failed: {e.error_code.value}")
        _display_error(e.to_error_response(trace_id=_get_trace_id()))
        return
    except Exception as e:
        err = create_error_from_exception(e, trace_id=_get_trace_id(), error_details=sys)
        st.sidebar.error("Model load failed")
        _display_error(err)
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
        return features_df, data

    df, data = user_input_features()

    # Display Input Parameters
    st.header("Specified Input Parameters")
    st.dataframe(df)
    st.write("---")

    # Prediction Section
    st.header("Prediction Results")

    if st.button("Predict"):
        trace_id = _get_trace_id()
        try:
            validation = data.validate()
            if not validation.is_valid:
                err = create_error_response(
                    ErrorCode.INPUT_INVALID_FORMAT,
                    trace_id=trace_id,
                    field="multiple",
                    value="; ".join(validation.errors),
                )
                _display_error(err)
                return

            # Make Prediction with detailed result
            detailed = pipeline.predict_detailed(df, transaction_id=trace_id)

            if not detailed.is_success():
                _display_error(detailed.error_detail)
                return

            # Keep original view-model behavior: prediction array + proba array
            prediction = [detailed.prediction]
            prediction_proba = [[detailed.legit_probability, detailed.fraud_probability]]

            # Display Prediction (original behavior)
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

            # Visualize Prediction Probabilities (original behavior)
            st.subheader("Prediction Probability Distribution")
            fig, ax = plt.subplots()
            ax.bar(proba_df.columns, proba_df.iloc[0], color=["green", "red"])
            ax.set_ylabel("Probability")
            ax.set_ylim(0, 1)
            ax.set_title("Fraud vs. Not Fraud Probability")
            st.pyplot(fig)

            # Risk Explanation (new addition)
            if detailed.risk_explanation and detailed.risk_explanation.top_factors:
                with st.expander("📊 Risk Factors Explanation", expanded=True):
                    st.subheader("Top Risk Factors")
                    for factor in detailed.risk_explanation.top_factors:
                        impact_icon = "🔴" if factor.get("impact") == "high" else ("🟠" if factor.get("impact") == "medium" else "🟡")
                        st.markdown(
                            f"- **{impact_icon} {factor.get('feature_name')}** "
                            f"→ {factor.get('label', factor.get('explanation', ''))} "
                            f"(contribution: {factor.get('importance', 0) * 100:.1f}%)"
                        )

            st.caption(f"Trace ID: {trace_id} · Model: v{detailed.model_version}")

        except AMLException as e:
            _display_error(e.to_error_response(trace_id=trace_id))
        except Exception as e:
            logging.error(f"Streamlit predict error: {e}", exc_info=True)
            err = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
            _display_error(err)

    st.write("---")
    st.markdown(
        """
        **Note:** This app is for demonstration purposes only. The predictions are based on a machine learning model.
        """
    )
    logging.info(f"Streamlit app execution completed")


if __name__ == "__main__":
    main()
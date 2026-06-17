import sys
import streamlit as st
import dill
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.config import get_config_service, ConfigService, reset_config_service
from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging


def _get_config_service() -> ConfigService:
    if "_aml_config_service" not in st.session_state:
        profile_override: Optional[str] = None
        project_root_override: Optional[str] = None
        st.session_state["_aml_config_service"] = get_config_service(
            profile=profile_override,
            project_root=project_root_override,
            force_reload=False,
        )
    return st.session_state["_aml_config_service"]


def _get_predict_pipeline() -> PredictionPipeline:
    if "_aml_predict_pipeline" not in st.session_state:
        config_service = _get_config_service()
        st.session_state["_aml_predict_pipeline"] = PredictionPipeline(
            config_service=config_service,
        )
    return st.session_state["_aml_predict_pipeline"]


def _render_sidebar_config_info() -> None:
    config_service = _get_config_service()
    meta = config_service.get_config_metadata()
    with st.sidebar.expander("⚙️ Configuration", expanded=False):
        st.write(f"**Profile**: `{meta['profile']}`")
        st.write(f"**Env**: `{config_service.config.env}`")
        st.write(f"**App Version**: `{config_service.config.app_version}`")
        st.write(f"**Config Source**: `{meta['config_path_source']}`")
        if meta["config_path"]:
            st.write(f"**Config File**: `{meta['config_path']}`")
        st.write(f"**Env Overrides**: `{meta['env_override_count']}` applied")

        artifacts_dir = config_service.config.prediction.default_artifacts_dir
        st.write(f"**Artifacts Dir**: `{artifacts_dir}`")

        thresholds = config_service.config.prediction.risk_thresholds
        st.write("**Risk Thresholds**:")
        for level_name, val in thresholds.items():
            st.write(f"  - {level_name}: `{val}`")

        if st.button("🔄 Reload Config", key="reload_cfg_btn"):
            reset_config_service()
            if "_aml_predict_pipeline" in st.session_state:
                del st.session_state["_aml_predict_pipeline"]
            if "_aml_config_service" in st.session_state:
                del st.session_state["_aml_config_service"]
            st.rerun()


def _render_config_debug_section() -> None:
    config_service = _get_config_service()
    with st.expander("🔍 Effective Config (Sanitized)", expanded=False):
        sanitized = config_service.get_effective_config(sanitized=True)
        st.json(sanitized)
        export_btn = st.button("📤 Export Effective Config", key="export_cfg_btn")
        if export_btn:
            try:
                path = config_service.export_effective_config(
                    sanitized=True,
                    include_metadata=True,
                )
                st.success(f"Config exported to: `{path}`")
            except Exception as e:
                st.error(f"Export failed: {e}")


def main():
    logging.info(f"Starting Streamlit App (via ConfigService)")

    config_service = _get_config_service()
    app_cfg = config_service.config

    st.title(f"{app_cfg.app_name}")
    st.caption(f"v{app_cfg.app_version}  |  Environment: **{config_service.profile.upper()}**")
    st.markdown(
        """
        This app predicts whether a given transaction is **fraudulent** or **non-fraudulent**.
        Please provide the required input features in the sidebar and click the **Predict** button.
        """
    )
    st.write("---")

    _render_sidebar_config_info()

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
            st.sidebar.error("❌ Input validation errors:")
            for err in validation.errors:
                st.sidebar.warning(f"- {err}")
        features_df = data.get_data_as_DataFrame()
        return features_df, validation

    df, validation = user_input_features()

    st.header("Specified Input Parameters")
    st.dataframe(df)
    if not validation.is_valid:
        st.error("⚠️ Some input fields have validation issues. Fix them in the sidebar before predicting.")
    st.write("---")

    st.header("Prediction Results")
    predict_pipeline = _get_predict_pipeline()

    pipeline_info_cols = st.columns(4)
    with pipeline_info_cols[0]:
        st.metric("Artifacts Dir", f"{predict_pipeline.artifacts_dir[:24]}..." if len(predict_pipeline.artifacts_dir) > 24 else predict_pipeline.artifacts_dir)
    with pipeline_info_cols[1]:
        st.metric("Model Loaded", "✅ Yes" if predict_pipeline._model is not None else "⏳ Lazy")
    with pipeline_info_cols[2]:
        st.metric("Preprocessor", "✅" if predict_pipeline._preprocessor is not None else "⏳")
    with pipeline_info_cols[3]:
        thresholds = predict_pipeline.risk_thresholds
        st.metric("Risk: Critical ≥", f"{thresholds.get('critical', 0.9):.0%}")

    predict_btn = st.button("🔮 Predict", disabled=(not validation.is_valid), type="primary")
    if predict_btn:
        if not validation.is_valid:
            st.error("Cannot predict: input validation failed.")
        else:
            with st.spinner("Running prediction..."):
                prediction = predict_pipeline.predict(df)
                prediction_proba = predict_pipeline.predict_proba(df)

            if not prediction.is_success():
                st.error(f"Prediction failed: `{prediction.error_detail.error_code.value}` - {prediction.error_detail.message}")
                with st.expander("Error Details", expanded=False):
                    st.json(prediction.error_detail.to_dict())
            else:
                st.subheader("Fraud Detector Class Labels")
                class_labels_df = pd.DataFrame({"Not Fraud": [0], "Fraud": [1]})
                class_labels_df.index = ["Class Labels"]
                st.dataframe(class_labels_df.T)

                st.subheader("Prediction of the Given Transaction")
                fraud_label = prediction.prediction == 1
                if fraud_label:
                    st.error("🚨 **Fraudulent Transaction**")
                else:
                    st.success("✅ **Non-Fraudulent Transaction**")

                st.info(
                    f"**Class**: `{prediction.class_label}`  |  "
                    f"**Risk Level**: `{prediction.risk_explanation.risk_level.value if prediction.risk_explanation else 'UNKNOWN'}`  |  "
                    f"**Model Version**: `{prediction.model_version}`"
                )

                st.subheader("Prediction Probabilities")
                proba_df = pd.DataFrame(prediction_proba, columns=["Not Fraud", "Fraud"])
                st.dataframe(proba_df.style.format("{:.4%}"))

                st.subheader("Prediction Probability Distribution")
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ["#28a745", "#dc3545"]
                bars = ax.bar(proba_df.columns, proba_df.iloc[0], color=colors, edgecolor="black")
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        f"{height:.2%}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                    )
                ax.set_ylabel("Probability")
                ax.set_ylim(0, 1.05)
                ax.set_title("Fraud vs. Not Fraud Probability", fontsize=12, fontweight="bold")
                ax.grid(axis="y", linestyle="--", alpha=0.5)
                st.pyplot(fig, clear_figure=True)

                thresholds = predict_pipeline.risk_thresholds
                st.subheader("Risk Level Breakdown")
                risk_cols = st.columns(4)
                with risk_cols[0]:
                    cr = thresholds.get("critical", 0.9)
                    st.metric("CRITICAL", f"≥ {cr:.0%}", delta=f"fraud_prob ≥ {cr}")
                with risk_cols[1]:
                    hi = thresholds.get("high", 0.7)
                    st.metric("HIGH", f"≥ {hi:.0%}")
                with risk_cols[2]:
                    md = thresholds.get("medium", 0.5)
                    st.metric("MEDIUM", f"≥ {md:.0%}")
                with risk_cols[3]:
                    st.metric("LOW", f"< {md:.0%}")

    st.write("---")

    _render_config_debug_section()

    st.markdown(
        f"""
        **Note:** This app is for demonstration purposes only.
        The predictions are based on a machine learning model loaded from `{predict_pipeline.artifacts_dir}`.
        """
    )
    logging.info(f"Streamlit app execution completed (profile={config_service.profile})")


if __name__ == "__main__":
    main()

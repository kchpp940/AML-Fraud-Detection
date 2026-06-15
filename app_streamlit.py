import streamlit as st
import dill
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
import pandas as pd
import matplotlib.pyplot as plt

def _render_explanation(result: dict):
    prediction = result["prediction"]
    prediction_label = result["prediction_label"]
    fraud_probability = result["fraud_probability"]
    top_factors = result.get("top_factors", [])

    st.subheader("预测结果")
    if prediction == 1:
        st.error(f"**{prediction_label}**")
    else:
        st.success(f"**{prediction_label}**")

    proba_df = pd.DataFrame({
        "类别": ["正常交易", "欺诈交易"],
        "概率": [1 - fraud_probability, fraud_probability],
    })
    st.dataframe(proba_df, hide_index=True)

    fig, ax = plt.subplots()
    colors = ["green", "red"]
    ax.bar(proba_df["类别"], proba_df["概率"], color=colors)
    ax.set_ylabel("概率")
    ax.set_title("风险概率分布")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    if not top_factors:
        st.info("暂无风险解释信息（feature_metadata.json 不存在，请重新训练模型以生成）")
        return

    st.subheader("风险因素分析")
    st.markdown(
        f"以下是影响本次风险判断的关键因素及其贡献度："
    )

    for i, factor in enumerate(top_factors, 1):
        label = factor["label"]
        value = factor["value"]
        contribution_pct = factor["contribution_pct"]
        direction = factor["direction"]
        description = factor["description"]

        direction_icon = "🔴" if direction == "high_risk" else "🟢"
        direction_text = "增加风险" if direction == "high_risk" else "降低风险"

        st.markdown(
            f"**{i}. {label}** — 当前值: `{value}`  \n"
            f"{direction_icon} 贡献度: **{contribution_pct:.1f}%**（{direction_text}）  \n"
            f"📝 {description}"
        )

        progress_val = min(contribution_pct / 100.0, 1.0)
        bar_color = "🔴" if direction == "high_risk" else "🟢"
        st.progress(progress_val)

    factor_labels = [f["label"] for f in top_factors]
    factor_pcts = [f["contribution_pct"] for f in top_factors]
    factor_colors = ["#e74c3c" if f["direction"] == "high_risk" else "#2ecc71" for f in top_factors]

    fig2, ax2 = plt.subplots()
    bars = ax2.barh(factor_labels, factor_pcts, color=factor_colors)
    ax2.set_xlabel("贡献度 (%)")
    ax2.set_title("风险因素贡献度")
    ax2.invert_yaxis()
    for bar, pct in zip(bars, factor_pcts):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{pct:.1f}%", va="center", fontsize=9)
    st.pyplot(fig2)


def main():
    logging.info(f"Starting Streamlit App")

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
        features_df = data.get_data_as_DataFrame()
        return features_df

    df = user_input_features()

    st.header("Specified Input Parameters")
    st.dataframe(df)
    st.write("---")

    st.header("Prediction Results")
    predict_pipeline = PredictionPipeline()

    if st.button("Predict"):
        result = predict_pipeline.predict_with_explanation(df)
        _render_explanation(result)

    st.write("---")
    st.markdown(
        """
        **Note:** This app is for demonstration purposes only. The predictions are based on a machine learning model.
        """
    )
    logging.info(f"Streamlit app execution completed")


if __name__ == "__main__":
    main()

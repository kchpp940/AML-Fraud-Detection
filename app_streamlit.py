import streamlit as st
import dill
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
import pandas as pd
import matplotlib.pyplot as plt

BATCH_COLUMNS = [
    "from_bank", "account", "to_bank", "account_1",
    "amount_received", "receiving_currency", "payment_currency",
    "payment_format", "day"
]


def _render_single_explanation(result: dict, sig_valid: bool, sig_msg: str, contract_version: str, training_sig: str):
    prediction = result["prediction"]
    prediction_label = result["prediction_label"]
    fraud_probability = result["fraud_probability"]
    top_factors = result.get("risk_explanation", {}).get("top_factors", [])
    process_status = result.get("process_status", "success")
    error_reason = result.get("error_reason")

    st.subheader("预测结果")
    if process_status == "failed":
        st.error(f"**预测失败**: {error_reason}")
        return
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
    st.markdown("以下是影响本次风险判断的关键因素及其贡献度：")

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


def _render_batch_explanation(full_result: dict, input_df: pd.DataFrame, sig_valid: bool, sig_msg: str):
    count = full_result["count"]
    fraud_count = full_result.get("fraud_count", 0)
    normal_count = full_result.get("normal_count", 0)
    fraud_rate = full_result.get("fraud_rate", 0.0)
    rows = full_result.get("rows", [])

    st.subheader("批量预测汇总")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("交易总数", count)
    col2.metric("欺诈数量", fraud_count, delta_color="inverse")
    col3.metric("正常数量", normal_count)
    col4.metric("欺诈率", f"{fraud_rate}%")

    summary_df = pd.DataFrame({
        "类别": ["正常交易", "欺诈交易"],
        "数量": [normal_count, fraud_count],
        "占比": [f"{normal_count / count * 100:.1f}%" if count > 0 else "0%",
                f"{fraud_count / count * 100:.1f}%" if count > 0 else "0%"],
    })
    st.dataframe(summary_df, hide_index=True)

    output_rows = []
    for i, res in enumerate(rows):
        orig_row = input_df.iloc[i].to_dict()
        row = {
            "行号": i + 1,
            "处理状态": "成功" if res.get("process_status") == "success" else "失败",
            "预测结果": res.get("prediction_label"),
            "欺诈概率": f"{res.get('fraud_probability', 0) * 100:.2f}%",
            "错误原因": res.get("error_reason"),
        }
        top_factors = res.get("risk_explanation", {}).get("top_factors", [])
        for j, factor in enumerate(top_factors, 1):
            row[f"风险因素{j}"] = f"{factor.get('label')}={factor.get('value')} ({factor.get('contribution_pct')}%)"
            row[f"因素{j}说明"] = factor.get("description")
        row.update(orig_row)
        output_rows.append(row)

    output_df = pd.DataFrame(output_rows)

    st.subheader("预测详情（含风险解释）")
    st.dataframe(output_df, use_container_width=True)

    csv = output_df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="⬇️ 下载完整结果 (CSV)",
        data=csv,
        file_name="batch_predictions_with_explanation.csv",
        mime="text/csv",
    )

    st.subheader("按预测结果筛选")
    filter_opt = st.selectbox("选择查看", ["全部", "仅欺诈交易", "仅正常交易", "仅失败行"])
    if filter_opt == "仅欺诈交易":
        filtered = output_df[(output_df["预测结果"] == "欺诈交易") & (output_df["处理状态"] == "成功")]
    elif filter_opt == "仅正常交易":
        filtered = output_df[(output_df["预测结果"] == "正常交易") & (output_df["处理状态"] == "成功")]
    elif filter_opt == "仅失败行":
        filtered = output_df[output_df["处理状态"] == "失败"]
    else:
        filtered = output_df
    st.dataframe(filtered, use_container_width=True)

    if fraud_count > 0:
        st.subheader("高风险交易 Top 5")
        risk_sorted = output_df.copy()
        risk_sorted["_proba_val"] = risk_sorted["欺诈概率"].str.replace("%", "").astype(float)
        top_risk = risk_sorted.nlargest(5, "_proba_val").drop(columns=["_proba_val"])
        st.dataframe(top_risk, use_container_width=True)


def main():
    logging.info(f"Starting Streamlit App")

    st.title("Anti-Money Laundering (AML) Fraud Detection")
    st.markdown(
        """
        This app predicts whether a given transaction is **fraudulent** or **non-fraudulent**.
        Please choose prediction mode in the sidebar and provide the required input features.
        """
    )
    st.write("---")

    mode = st.sidebar.radio("预测模式", ["单条预测", "批量预测"], index=0)
    predict_pipeline = PredictionPipeline()

    if mode == "单条预测":
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

        if st.button("Predict"):
            full_result = predict_pipeline.predict_with_explanation(df, top_n=5)
            sig_valid = full_result.get("signature_valid", True)
            sig_msg = full_result.get("signature_message", "")
            contract_version = full_result.get("contract_version", "1.0")
            training_sig = full_result.get("training_signature", "")

            if not sig_valid:
                st.warning(f"⚠️ {sig_msg}")

            single_result = full_result.get("row", {})
            _render_single_explanation(single_result, sig_valid, sig_msg, contract_version, training_sig)

            if training_sig:
                st.caption(f"契约版本: v{contract_version} | 训练签名: {training_sig}")

    else:
        st.header("批量交易预测")
        st.markdown(
            f"上传 CSV 文件，需包含以下列: `{', '.join(BATCH_COLUMNS)}`"
        )

        uploaded_file = st.file_uploader("选择 CSV 文件", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"CSV 文件解析失败: {str(e)}")
                return

            missing_cols = [c for c in BATCH_COLUMNS if c not in df.columns]
            if missing_cols:
                st.error(f"CSV 缺少必需的列: {', '.join(missing_cols)}")
                return

            st.subheader("上传数据预览")
            st.dataframe(df.head(10))
            st.caption(f"共 {len(df)} 条交易记录")

            if st.button("开始批量预测"):
                predict_df = df[BATCH_COLUMNS].copy()
                full_result = predict_pipeline.predict_with_explanation(predict_df, top_n=5)

                sig_valid = full_result.get("signature_valid", True)
                sig_msg = full_result.get("signature_message", "")

                if not sig_valid:
                    st.warning(f"⚠️ {sig_msg}")

                _render_batch_explanation(full_result, df, sig_valid, sig_msg)

                training_sig = full_result.get("training_signature", "")
                contract_version = full_result.get("contract_version", "1.0")
                if training_sig:
                    st.caption(f"契约版本: v{contract_version} | 训练签名: {training_sig}")

    st.write("---")
    st.markdown(
        """
        **Note:** This app is for demonstration purposes only. The predictions are based on a machine learning model.
        """
    )
    logging.info(f"Streamlit app execution completed")


if __name__ == "__main__":
    main()

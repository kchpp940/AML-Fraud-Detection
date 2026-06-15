import io
import pandas as pd
from flask import Flask, request, render_template, send_file, Response
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)
app = application

BATCH_COLUMNS = [
    "from_bank", "account", "to_bank", "account_1",
    "amount_received", "receiving_currency", "payment_currency",
    "payment_format", "day"
]


def _get_single_result(full_result: dict) -> dict:
    if full_result.get("is_batch") or len(full_result.get("results", [])) > 1:
        return full_result.get("results", [{}])[0]
    return full_result.get("results", [{}])[0]


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            from_bank = request.form.get("from_bank"),
            account = request.form.get("account"),
            to_bank = request.form.get("to_bank"),
            account_1 = request.form.get("account_1"),
            amount_received =  request.form.get("amount_received"),
            receiving_currency = request.form.get("receiving_currency"),
            payment_currency = request.form.get("payment_currency"),
            payment_format = request.form.get("payment_format"),
            day = request.form.get("day")
        )
        pred_df = data.get_data_as_DataFrame()
        print(pred_df)

        predict_pipeline = PredictionPipeline()
        full_result = predict_pipeline.predict_with_explanation(pred_df)
        single_result = _get_single_result(full_result)

        return render_template(
            "home.html",
            result=single_result,
            signature_valid=full_result.get("signature_valid", True),
            signature_message=full_result.get("signature_message", ""),
            contract_version=full_result.get("contract_version", "1.0"),
            training_signature=full_result.get("training_signature", ""),
        )


@app.route("/batchpredict", methods=["GET", "POST"])
def batch_predict():
    if request.method == "GET":
        return render_template("batch.html", columns=BATCH_COLUMNS)
    else:
        if "file" not in request.files:
            return render_template(
                "batch.html",
                columns=BATCH_COLUMNS,
                error="请选择要上传的 CSV 文件"
            )

        file = request.files["file"]
        if file.filename == "":
            return render_template(
                "batch.html",
                columns=BATCH_COLUMNS,
                error="未选择文件"
            )

        try:
            df = pd.read_csv(file)
        except Exception as e:
            return render_template(
                "batch.html",
                columns=BATCH_COLUMNS,
                error=f"CSV 文件解析失败: {str(e)}"
            )

        missing_cols = [c for c in BATCH_COLUMNS if c not in df.columns]
        if missing_cols:
            return render_template(
                "batch.html",
                columns=BATCH_COLUMNS,
                error=f"CSV 缺少必需的列: {', '.join(missing_cols)}"
            )

        predict_df = df[BATCH_COLUMNS].copy()
        predict_pipeline = PredictionPipeline()
        full_result = predict_pipeline.predict_with_explanation(predict_df)

        rows = []
        for i, res in enumerate(full_result.get("results", [])):
            orig_row = df.iloc[i].to_dict()
            row = {
                "row_index": i,
                "prediction": res.get("prediction"),
                "prediction_label": res.get("prediction_label"),
                "fraud_probability": res.get("fraud_probability"),
            }
            top_factors = res.get("top_factors", [])
            for j, factor in enumerate(top_factors, 1):
                row[f"factor_{j}_label"] = factor.get("label")
                row[f"factor_{j}_value"] = factor.get("value")
                row[f"factor_{j}_contribution_pct"] = factor.get("contribution_pct")
                row[f"factor_{j}_direction"] = factor.get("direction")
                row[f"factor_{j}_description"] = factor.get("description")
            row.update(orig_row)
            rows.append(row)

        output_df = pd.DataFrame(rows)
        csv_buffer = io.StringIO()
        output_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        csv_content = csv_buffer.getvalue()

        return render_template(
            "batch.html",
            columns=BATCH_COLUMNS,
            result=full_result,
            rows=rows,
            top_n=5,
            csv_download=csv_content,
        )


@app.route("/download_batch", methods=["POST"])
def download_batch():
    csv_content = request.form.get("csv_content", "")
    return Response(
        csv_content,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=batch_predictions_with_explanation.csv"}
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

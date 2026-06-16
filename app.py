import io
from flask import Flask, request, render_template
import pandas as pd

from aml_fraud_detector.pipeline.prediction_pipeline import CustomData
from aml_fraud_detector.presentation.response_builder import ResponseBuilder

application = Flask(__name__)
app = application
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

_builder = ResponseBuilder()


def _custom_data_to_dict(data: CustomData) -> dict:
    return {
        "from_bank": data.from_bank,
        "account": data.account,
        "to_bank": data.to_bank,
        "account_1": data.account_1,
        "amount_received": data.amount_received,
        "receiving_currency": data.receiving_currency,
        "payment_currency": data.payment_currency,
        "payment_format": data.payment_format,
        "day": data.day,
    }


@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# 单条预测
# ---------------------------------------------------------------------------
@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        vm = _builder.build_validation_only()
        display = ResponseBuilder.flatten_for_display(vm)
        return render_template("home.html", display=display)

    data = CustomData(
        from_bank=request.form.get("from_bank"),
        account=request.form.get("account"),
        to_bank=request.form.get("to_bank"),
        account_1=request.form.get("account_1"),
        amount_received=request.form.get("amount_received"),
        receiving_currency=request.form.get("receiving_currency"),
        payment_currency=request.form.get("payment_currency"),
        payment_format=request.form.get("payment_format"),
        day=request.form.get("day", "0"),
    )
    vm = _builder.build_single(
        data.to_dict() if hasattr(data, "to_dict") else _custom_data_to_dict(data)
    )
    display = ResponseBuilder.flatten_for_display(vm)
    return render_template("home.html", display=display)


# ---------------------------------------------------------------------------
# 批量预测
# ---------------------------------------------------------------------------
@app.route("/batchprediction", methods=["GET", "POST"])
def batch_prediction():
    if request.method == "GET":
        vm = _builder.build_validation_only()
        display = ResponseBuilder.flatten_for_display(vm)
        return render_template("batch.html", display=display)

    upload = request.files.get("csv_file")
    if upload is None or upload.filename is None or upload.filename == "":
        vm = _builder.build_validation_only()
        display = ResponseBuilder.flatten_for_display(vm)
        return render_template(
            "batch.html",
            display=display,
            batch_error="请选择要上传的 CSV 文件",
        )

    try:
        raw = upload.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        df = pd.read_csv(io.StringIO(raw))
    except Exception as exc:
        vm = _builder.build_validation_only()
        display = ResponseBuilder.flatten_for_display(vm)
        return render_template(
            "batch.html",
            display=display,
            batch_error=f"CSV 解析失败：{exc}",
        )

    vm = _builder.build_batch(df)
    display = ResponseBuilder.flatten_for_display(vm)
    batch_df = ResponseBuilder.batch_to_display_dataframe(vm)
    return render_template(
        "batch.html",
        display=display,
        batch_df=batch_df,
        input_count=len(df),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

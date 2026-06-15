from flask import Flask, request, render_template
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData
from aml_fraud_detector.presentation.response_builder import ResponseBuilder

application = Flask(__name__)
app = application

_builder = ResponseBuilder()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
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
        vm = _builder.build_single(data.to_dict() if hasattr(data, "to_dict") else _custom_data_to_dict(data))
        display = ResponseBuilder.flatten_for_display(vm)
        return render_template("home.html", display=display)


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

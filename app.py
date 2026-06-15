from flask import Flask, request, render_template
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.presentation.response_builder import ResponseBuilder

application = Flask(__name__)
app = application

_pipeline = PredictionPipeline()
_builder = ResponseBuilder(_pipeline)


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
        response = _builder.build_single_response(data.to_dict())
        display = ResponseBuilder.extract_display_fields(response)
        return render_template("home.html", display=display)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

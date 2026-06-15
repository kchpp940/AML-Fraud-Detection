from flask import Flask, request, render_template, jsonify
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/model-info")
def model_info():
    pipeline = PredictionPipeline()
    return jsonify(pipeline.model_metadata)


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        pipeline = PredictionPipeline()
        return render_template("home.html", model_metadata=pipeline.model_metadata)
    else:
        data = CustomData(
            from_bank=_safe_int(request.form.get("from_bank"), 0),
            account=request.form.get("account") or "",
            to_bank=_safe_int(request.form.get("to_bank"), 0),
            account_1=request.form.get("account_1") or "",
            amount_received=_safe_float(request.form.get("amount_received"), 0.0),
            receiving_currency=request.form.get("receiving_currency") or "",
            payment_currency=request.form.get("payment_currency") or "",
            payment_format=request.form.get("payment_format") or "",
            day=request.form.get("day") or "",
        )
        pipeline = PredictionPipeline()
        result = pipeline.predict_single(data, explain=True)
        results = result.prediction
        return render_template(
            "home.html",
            results=results,
            model_metadata=pipeline.model_metadata,
            result=result,
        )


def _safe_int(value, default):
    try:
        return int(float(value)) if value not in (None, "") else default
    except (ValueError, TypeError):
        return default


def _safe_float(value, default):
    try:
        return float(value) if value not in (None, "") else default
    except (ValueError, TypeError):
        return default


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

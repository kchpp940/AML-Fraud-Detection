from flask import Flask, request, render_template
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
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

        context = {
            "result": result,
            "results": result.prediction,
            "class_label": result.class_label,
            "fraud_probability": f"{result.fraud_probability * 100:.2f}%",
            "legit_probability": f"{result.legit_probability * 100:.2f}%",
        }
        if result.explanation is not None:
            context["risk_level"] = result.explanation.risk_level
            context["summary_text"] = result.explanation.summary_text
            context["top_contributors"] = result.explanation.top_contributors
        return render_template("home.html", **context)


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

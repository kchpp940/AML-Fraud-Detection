from flask import Flask, request, render_template, jsonify
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)
app = application

_PREDICTION_PIPELINE = PredictionPipeline()


def _model_context():
    return {
        "model_metadata": _PREDICTION_PIPELINE.model_metadata,
        "validation_ok": _PREDICTION_PIPELINE.validation_ok,
        "validation_warnings": _PREDICTION_PIPELINE.validation_warnings,
        "validation_errors": _PREDICTION_PIPELINE.validation_errors,
    }


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/model-info")
def model_info():
    ctx = _model_context()
    return jsonify({
        "validation_ok": ctx["validation_ok"],
        "validation_warnings": ctx["validation_warnings"],
        "validation_errors": ctx["validation_errors"],
        "metadata": ctx["model_metadata"],
    })

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    ctx = _model_context()
    if request.method == "GET":
        return render_template("home.html", results=None, **ctx)
    else:
        data = CustomData(
            from_bank = request.form.get("from_bank"),
            account = request.form.get("account"),
            to_bank = request.form.get("to_bank"),
            account_1 = request.form.get("account_1"),
            amount_received =  request.form.get("amount_received"),
            receiving_currency = request.form.get("receiving_currency"),
            payment_currency = request.form.get("payment_currency"),
            payment_format = request.form.get("payment_format")
        )
        pred_df = data.get_data_as_DataFrame()
        print(pred_df)

        results = _PREDICTION_PIPELINE.predict(pred_df)
        return render_template("home.html", results=results[0], **ctx)
    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

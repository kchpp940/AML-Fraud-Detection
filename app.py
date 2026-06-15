from flask import Flask, request, render_template, jsonify
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.utils.main_utils import load_model_metadata

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/model-info")
def model_info():
    metadata = load_model_metadata()
    return jsonify(metadata)

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        metadata = load_model_metadata()
        return render_template("home.html", model_metadata=metadata)
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

        predict_pipeline = PredictionPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template("home.html", results=results[0], model_metadata=predict_pipeline.model_metadata)
    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
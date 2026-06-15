from flask import Flask, request, render_template
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.logger import logging

application = Flask(__name__)
app = application

predict_pipeline = PredictionPipeline()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData.from_flask_request(request.form)
        pred_df = data.get_aligned_DataFrame()
        logging.info(f"Flask received aligned features: {pred_df.columns.tolist()}")
        print(f"Aligned features: {pred_df.to_dict(orient='records')}")

        results = predict_pipeline.predict(pred_df)
        return render_template("home.html", results=results[0])
    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
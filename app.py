from flask import Flask, request, render_template
from aml_fraud_detector.pipeline.prediction_pipeline import (
    CustomData, PredictionPipeline, InputValidationError
)
from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    try:
        data = CustomData(
            from_bank=request.form.get("from_bank"),
            account=request.form.get("account"),
            to_bank=request.form.get("to_bank"),
            account_1=request.form.get("account_1"),
            amount_received=request.form.get("amount_received"),
            receiving_currency=request.form.get("receiving_currency"),
            payment_currency=request.form.get("payment_currency"),
            payment_format=request.form.get("payment_format"),
            day=request.form.get("day"),
        )
        pred_df = data.get_data_as_DataFrame()
        logging.info(f"Prepared input DataFrame:\n{pred_df.to_string()}")

        predict_pipeline = PredictionPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template("home.html", results=results[0])

    except InputValidationError as e:
        logging.warning(f"Input validation failed: {e}")
        return render_template(
            "home.html",
            error=str(e),
            form_data=request.form.to_dict(),
        )

    except CustomerException as e:
        logging.error(f"Prediction pipeline error: {e}")
        return render_template(
            "home.html",
            error=f"预测服务异常，请稍后重试或联系管理员。详情: {e.error_message}",
            form_data=request.form.to_dict(),
        )

    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return render_template(
            "home.html",
            error=f"服务器内部错误，请稍后重试。",
            form_data=request.form.to_dict(),
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
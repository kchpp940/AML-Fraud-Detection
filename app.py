import io
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_file, Response
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline, REQUIRED_FIELDS

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
        data = CustomData(
            from_bank=request.form.get("from_bank"),
            account=request.form.get("account"),
            to_bank=request.form.get("to_bank"),
            account_1=request.form.get("account_1"),
            amount_received=request.form.get("amount_received"),
            receiving_currency=request.form.get("receiving_currency"),
            payment_currency=request.form.get("payment_currency"),
            payment_format=request.form.get("payment_format"),
            day=request.form.get("day", "Monday")
        )
        pred_df = data.get_data_as_DataFrame()
        print(pred_df)

        results = predict_pipeline.predict(pred_df)
        return render_template("home.html", results=results[0])


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        custom_data = CustomData(
            from_bank=data.get("from_bank"),
            account=data.get("account"),
            to_bank=data.get("to_bank"),
            account_1=data.get("account_1"),
            amount_received=data.get("amount_received"),
            receiving_currency=data.get("receiving_currency"),
            payment_currency=data.get("payment_currency"),
            payment_format=data.get("payment_format"),
            day=data.get("day", "Monday")
        )
        pred_df = custom_data.get_data_as_DataFrame()
        prediction = predict_pipeline.predict(pred_df)
        prediction_proba = predict_pipeline.predict_proba(pred_df)
        
        return jsonify({
            "success": True,
            "prediction_label": int(prediction[0]),
            "fraud_probability": float(prediction_proba[0][1]),
            "is_fraud": bool(prediction[0] == 1)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route("/api/predict/batch", methods=["POST"])
def api_predict_batch():
    try:
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "未找到上传的文件，请使用 'file' 字段上传 CSV 文件"
            }), 400
        
        file = request.files["file"]
        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "文件名为空"
            }), 400
        
        if not file.filename.lower().endswith(".csv"):
            return jsonify({
                "success": False,
                "error": "只支持 CSV 格式文件"
            }), 400
        
        file_content = file.read().decode("utf-8")
        input_df = pd.read_csv(io.StringIO(file_content))
        
        if len(input_df) == 0:
            return jsonify({
                "success": False,
                "error": "CSV 文件为空"
            }), 400
        
        result_df = predict_pipeline.predict_batch(input_df)
        
        format_type = request.args.get("format", "json")
        
        if format_type == "csv":
            output = io.StringIO()
            result_df.to_csv(output, index=False, encoding="utf-8-sig")
            output.seek(0)
            return Response(
                output.getvalue(),
                mimetype="text/csv; charset=utf-8",
                headers={"Content-Disposition": "attachment; filename=prediction_results.csv"}
            )
        
        result_records = result_df.to_dict(orient="records")
        success_count = result_df["prediction_label"].notna().sum()
        fail_count = result_df["prediction_label"].isna().sum()
        fraud_count = (result_df["prediction_label"] == 1).sum()
        
        return jsonify({
            "success": True,
            "total_rows": len(result_df),
            "success_rows": int(success_count),
            "failed_rows": int(fail_count),
            "fraud_count": int(fraud_count),
            "required_fields": list(REQUIRED_FIELDS.keys()),
            "results": result_records
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/fields", methods=["GET"])
def api_fields():
    return jsonify({
        "success": True,
        "required_fields": [
            {
                "name": field,
                "type": dtype.__name__,
                "description": {
                    "from_bank": "汇出银行ID",
                    "account": "汇款人账号",
                    "to_bank": "汇入银行ID",
                    "account_1": "收款人账号",
                    "amount_received": "到账金额",
                    "receiving_currency": "收款币种",
                    "payment_currency": "付款币种",
                    "payment_format": "付款方式",
                    "day": "交易星期"
                }.get(field, "")
            }
            for field, dtype in REQUIRED_FIELDS.items()
        ]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
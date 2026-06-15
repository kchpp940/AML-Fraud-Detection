import io
import base64
import pandas as pd
from flask import Flask, request, render_template, jsonify, Response
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline

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


@app.route("/batch", methods=["GET", "POST"])
def batch_prediction():
    if request.method == "GET":
        schema = predict_pipeline.get_schema_info()
        return render_template("batch.html", schema=schema)

    if "file" not in request.files:
        schema = predict_pipeline.get_schema_info()
        return render_template("batch.html", schema=schema, error="未找到上传的文件")

    file = request.files["file"]
    if file.filename == "":
        schema = predict_pipeline.get_schema_info()
        return render_template("batch.html", schema=schema, error="文件名为空")

    if not file.filename.lower().endswith(".csv"):
        schema = predict_pipeline.get_schema_info()
        return render_template("batch.html", schema=schema, error="只支持 CSV 格式文件")

    try:
        file_content = file.read().decode("utf-8")
        input_df = pd.read_csv(io.StringIO(file_content))

        if len(input_df) == 0:
            schema = predict_pipeline.get_schema_info()
            return render_template("batch.html", schema=schema, error="CSV 文件为空")

        result_df = predict_pipeline.predict_batch(input_df)

        total = len(result_df)
        success_count = int(result_df["prediction_label"].notna().sum())
        fail_count = int(result_df["prediction_label"].isna().sum())
        fraud_count = int((result_df["prediction_label"] == 1).sum())

        csv_output = io.StringIO()
        result_df.to_csv(csv_output, index=False, encoding="utf-8-sig")
        csv_b64 = base64.b64encode(csv_output.getvalue().encode("utf-8-sig")).decode("ascii")

        display_df = result_df.copy()
        table_html = display_df.to_html(
            classes="table table-striped table-bordered table-sm",
            index=False,
            na_rep="-",
            max_rows=200,
            escape=False
        )

        return render_template(
            "batch.html",
            schema=predict_pipeline.get_schema_info(),
            table_html=table_html,
            csv_b64=csv_b64,
            stats={
                "total": total,
                "success": success_count,
                "failed": fail_count,
                "fraud": fraud_count,
            }
        )

    except Exception as e:
        schema = predict_pipeline.get_schema_info()
        return render_template("batch.html", schema=schema, error=str(e))


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
        success_count = int(result_df["prediction_label"].notna().sum())
        fail_count = int(result_df["prediction_label"].isna().sum())
        fraud_count = int((result_df["prediction_label"] == 1).sum())
        
        schema = predict_pipeline.get_schema_info()

        return jsonify({
            "success": True,
            "total_rows": len(result_df),
            "success_rows": success_count,
            "failed_rows": fail_count,
            "fraud_count": fraud_count,
            "schema": schema,
            "results": result_records
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/schema", methods=["GET"])
def api_schema():
    return jsonify({
        "success": True,
        "schema": predict_pipeline.get_schema_info()
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

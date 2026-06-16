import sys
import uuid
import json
import pandas as pd
from flask import Flask, request, render_template, jsonify, make_response

from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.exception import (
    AMLException,
    InputValidationException,
    create_error_from_exception,
    create_error_response,
    UnifiedErrorResponse,
    ErrorDetail,
)
from aml_fraud_detector.constants import (
    ErrorCode,
    HTTP_STATUS_CODES,
    ERROR_CATEGORY_DISPLAY,
)
from aml_fraud_detector.entity import (
    PredictionResult,
    BatchPredictionResult,
    UnifiedPredictionResponse,
    ValidationStatus,
)
from aml_fraud_detector.logger import logging

application = Flask(__name__)
app = application

_predict_pipeline = None


def _get_pipeline():
    global _predict_pipeline
    if _predict_pipeline is None:
        _predict_pipeline = PredictionPipeline()
    return _predict_pipeline


def _get_trace_id():
    return str(uuid.uuid4())


def _error_to_display(error_detail, trace_id):
    if isinstance(error_detail, UnifiedErrorResponse):
        return error_detail.to_user_display()
    err = UnifiedErrorResponse(
        success=False,
        status="error",
        error=error_detail,
        http_status=HTTP_STATUS_CODES.get(error_detail.error_category, 500),
        trace_id=trace_id,
    )
    return err.to_user_display()


def _build_error_flask_response(error_resp):
    resp = make_response(jsonify(error_resp.to_dict()), error_resp.http_status)
    resp.headers["X-Trace-Id"] = error_resp.trace_id
    return resp


@app.errorhandler(AMLException)
def _aml_exception_handler(e):
    trace_id = _get_trace_id()
    logging.error(f"AMLException: [{e.error_code.value}] {e.message}")
    err = e.to_error_response(trace_id=trace_id)
    return _build_error_flask_response(err)


@app.errorhandler(Exception)
def _generic_exception_handler(e):
    trace_id = _get_trace_id()
    logging.error(f"Unexpected exception: {e}", exc_info=True)
    err = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
    return _build_error_flask_response(err)

# Route for a home page
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    trace_id = _get_trace_id()
    if request.method == "GET":
        return render_template("home.html", results=None, error=None, detail=None, trace_id=trace_id)
    else:
        try:
            data = CustomData(
                from_bank = request.form.get("from_bank"),
                account = request.form.get("account"),
                to_bank = request.form.get("to_bank"),
                account_1 = request.form.get("account_1"),
                amount_received =  request.form.get("amount_received"),
                receiving_currency = request.form.get("receiving_currency"),
                payment_currency = request.form.get("payment_currency"),
                payment_format = request.form.get("payment_format"),
                day = request.form.get("day", "Monday")
            )
            validation = data.validate()
            if not validation.is_valid:
                err_resp = create_error_response(
                    ErrorCode.INPUT_INVALID_FORMAT,
                    trace_id=trace_id,
                    field="multiple",
                    value="; ".join(validation.errors),
                )
                return render_template(
                    "home.html",
                    results=None,
                    error=_error_to_display(err_resp, trace_id),
                    detail=None,
                    trace_id=trace_id,
                )

            pred_df = data.get_data_as_DataFrame()
            print(pred_df)

            predict_pipeline = _get_pipeline()
            detailed = predict_pipeline.predict_detailed(pred_df, transaction_id=trace_id)

            if not detailed.is_success():
                return render_template(
                    "home.html",
                    results=None,
                    error=_error_to_display(detailed.error_detail, trace_id),
                    detail=None,
                    trace_id=trace_id,
                )

            results = detailed.prediction
            detail = {
                "class_label": detailed.class_label,
                "fraud_probability": detailed.fraud_probability,
                "legit_probability": detailed.legit_probability,
                "risk_level": detailed.risk_level.value,
                "model_version": detailed.model_version,
                "transaction_id": trace_id,
                "top_factors": [
                    {"feature_name": f["feature_name"], "label": f.get("label", ""), "importance": f.get("importance", 0)}
                    for f in (detailed.risk_explanation.top_factors if detailed.risk_explanation else [])
                ],
            }
            return render_template("home.html", results=results, error=None, detail=detail, trace_id=trace_id)

        except AMLException as e:
            err_resp = e.to_error_response(trace_id=trace_id)
            return render_template(
                "home.html",
                results=None,
                error=_error_to_display(err_resp, trace_id),
                detail=None,
                trace_id=trace_id,
            )
        except Exception as e:
            err_resp = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
            return render_template(
                "home.html",
                results=None,
                error=_error_to_display(err_resp, trace_id),
                detail=None,
                trace_id=trace_id,
            )
    
    
@app.route("/health", methods=["GET"])
def health_check():
    trace_id = _get_trace_id()
    try:
        pipeline = _get_pipeline()
        validation = pipeline.validate_artifacts(verify_digests=False)
        mv_info = pipeline.get_model_version_info()
        resp_data = {
            "status": "ok",
            "trace_id": trace_id,
            "artifacts_valid": validation.is_valid,
            "model_version": mv_info.model_version,
            "model_name": mv_info.model_name,
            "artifact_path": mv_info.artifact_path,
        }
        resp = make_response(jsonify(resp_data), 200)
        resp.headers["X-Trace-Id"] = trace_id
        return resp
    except AMLException as e:
        err = e.to_error_response(trace_id=trace_id)
        resp = _build_error_flask_response(err)
        resp_data = json.loads(resp.data)
        resp_data["status"] = "degraded"
        resp.set_data(json.dumps(resp_data))
        return resp


@app.route("/api/version", methods=["GET"])
def api_version():
    trace_id = _get_trace_id()
    pipeline = _get_pipeline()
    mv_info = pipeline.get_model_version_info()
    resp_data = {
        "trace_id": trace_id,
        "success": True,
        "model_version": mv_info.model_version,
        "model_name": mv_info.model_name,
        "training_time": mv_info.training_time,
        "selection_metric": mv_info.selection_metric,
        "best_metric_value": mv_info.best_metric_value,
        "feature_schema_version": mv_info.feature_schema_version,
        "artifact_path": mv_info.artifact_path,
    }
    resp = make_response(jsonify(resp_data), 200)
    resp.headers["X-Trace-Id"] = trace_id
    return resp


@app.route("/api/predict", methods=["POST"])
def api_predict():
    trace_id = _get_trace_id()
    try:
        req_data = request.get_json(silent=True)
        if req_data is None:
            raise InputValidationException(
                ErrorCode.INPUT_INVALID_FORMAT,
                error_details=sys,
                field="request_body",
                value="expected JSON",
            )

        required_fields = ["from_bank", "to_bank", "amount_received"]
        missing = [f for f in required_fields if req_data.get(f) is None]
        if missing:
            raise InputValidationException(
                ErrorCode.INPUT_MISSING_FIELD,
                error_details=sys,
                field=", ".join(missing),
            )

        data = CustomData(
            from_bank=req_data.get("from_bank"),
            account=req_data.get("account", ""),
            to_bank=req_data.get("to_bank"),
            account_1=req_data.get("account_1", ""),
            amount_received=req_data.get("amount_received"),
            receiving_currency=req_data.get("receiving_currency", ""),
            payment_currency=req_data.get("payment_currency", ""),
            payment_format=req_data.get("payment_format", ""),
            day=req_data.get("day", "Monday"),
        )
        validation = data.validate()
        if not validation.is_valid:
            raise InputValidationException(
                ErrorCode.INPUT_INVALID_FORMAT,
                error_details=sys,
                field="multiple",
                value="; ".join(validation.errors),
            )

        pred_df = data.get_data_as_DataFrame()
        pipeline = _get_pipeline()
        result = pipeline.predict_detailed(pred_df, transaction_id=trace_id)

        if not result.is_success():
            err = create_error_response(
                result.error_detail.error_code,
                trace_id=trace_id,
                **result.error_detail.context,
            )
            return _build_error_flask_response(err)

        unified = UnifiedPredictionResponse(
            model_version=pipeline.get_model_version_info(),
            validation=ValidationStatus(is_valid=True),
            single_prediction=result,
        )
        resp = make_response(jsonify(unified.to_dict()), 200)
        resp.headers["X-Trace-Id"] = trace_id
        return resp

    except AMLException as e:
        err = e.to_error_response(trace_id=trace_id)
        return _build_error_flask_response(err)
    except Exception as e:
        err = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
        return _build_error_flask_response(err)


@app.route("/batch", methods=["GET", "POST"])
def batch_predict_page():
    trace_id = _get_trace_id()
    if request.method == "GET":
        return render_template("batch.html", results=None, error=None, detail=None, trace_id=trace_id)

    try:
        if "file" not in request.files:
            err_resp = create_error_response(
                ErrorCode.INPUT_MISSING_FIELD,
                trace_id=trace_id,
                field="file",
            )
            return render_template(
                "batch.html",
                results=None,
                error=_error_to_display(err_resp, trace_id),
                detail=None,
                trace_id=trace_id,
            )

        file = request.files["file"]
        if file.filename == "":
            err_resp = create_error_response(
                ErrorCode.INPUT_EMPTY_VALUE,
                trace_id=trace_id,
                field="file",
            )
            return render_template(
                "batch.html",
                results=None,
                error=_error_to_display(err_resp, trace_id),
                detail=None,
                trace_id=trace_id,
            )

        df = pd.read_csv(file)
        tids = None
        if "transaction_id" in df.columns:
            tids = df["transaction_id"].astype(str).tolist()
            df = df.drop(columns=["transaction_id"])
        for col in ["from_bank", "to_bank"]:
            if col in df.columns:
                df[col] = df[col].astype("object")

        pipeline = _get_pipeline()
        br = pipeline.predict_batch(df)
        if tids and len(tids) == len(br.predictions):
            for i, p in enumerate(br.predictions):
                if p.transaction_id is None or str(p.transaction_id).isdigit():
                    p.transaction_id = tids[i]

        if not br.is_success():
            return render_template(
                "batch.html",
                results=None,
                error=_error_to_display(br.error_detail, trace_id),
                detail=None,
                trace_id=trace_id,
            )

        rows = []
        for p in br.predictions:
            rows.append({
                "transaction_id": p.transaction_id,
                "prediction": p.class_label,
                "fraud_probability": f"{p.fraud_probability * 100:.2f}%",
                "risk_level": p.risk_level.value,
            })

        detail = {
            "total": br.total_transactions,
            "fraud_count": br.fraud_count,
            "fraud_rate": f"{br.fraud_rate * 100:.2f}%",
            "overall_risk": br.overall_risk_level.value,
        }
        return render_template(
            "batch.html",
            results=rows,
            error=None,
            detail=detail,
            trace_id=trace_id,
        )

    except AMLException as e:
        err_resp = e.to_error_response(trace_id=trace_id)
        return render_template(
            "batch.html",
            results=None,
            error=_error_to_display(err_resp, trace_id),
            detail=None,
            trace_id=trace_id,
        )
    except Exception as e:
        err_resp = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
        return render_template(
            "batch.html",
            results=None,
            error=_error_to_display(err_resp, trace_id),
            detail=None,
            trace_id=trace_id,
        )


@app.route("/api/predict_batch", methods=["POST"])
def api_predict_batch():
    trace_id = _get_trace_id()
    try:
        req_data = request.get_json(silent=True)
        if req_data is None:
            raise InputValidationException(
                ErrorCode.INPUT_INVALID_FORMAT,
                error_details=sys,
                field="request_body",
                value="expected JSON",
            )

        if "transactions" not in req_data:
            raise InputValidationException(
                ErrorCode.INPUT_MISSING_FIELD,
                error_details=sys,
                field="transactions",
            )

        transactions = req_data["transactions"]
        if not isinstance(transactions, list) or len(transactions) == 0:
            raise InputValidationException(
                ErrorCode.INPUT_EMPTY_VALUE,
                error_details=sys,
                field="transactions",
            )

        df = pd.DataFrame(transactions)
        tids = None
        if "transaction_id" in df.columns:
            tids = df["transaction_id"].astype(str).tolist()
            df = df.drop(columns=["transaction_id"])
        for col in ["from_bank", "to_bank"]:
            if col in df.columns:
                df[col] = df[col].astype("object")

        pipeline = _get_pipeline()
        br = pipeline.predict_batch(df)
        if tids and len(tids) == len(br.predictions):
            for i, p in enumerate(br.predictions):
                if p.transaction_id is None or str(p.transaction_id).isdigit():
                    p.transaction_id = tids[i]

        if not br.is_success():
            err = create_error_response(
                br.error_detail.error_code,
                trace_id=trace_id,
                **br.error_detail.context,
            )
            return _build_error_flask_response(err)

        unified = UnifiedPredictionResponse(
            model_version=pipeline.get_model_version_info(),
            validation=ValidationStatus(is_valid=True),
            batch_prediction=br,
        )
        resp = make_response(jsonify(unified.to_dict()), 200)
        resp.headers["X-Trace-Id"] = trace_id
        return resp

    except AMLException as e:
        err = e.to_error_response(trace_id=trace_id)
        return _build_error_flask_response(err)
    except Exception as e:
        err = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
        return _build_error_flask_response(err)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
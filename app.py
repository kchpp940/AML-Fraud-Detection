import sys
import os
import uuid
from flask import Flask, request, render_template, jsonify, make_response
from flask_cors import CORS

from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.exception import (
    AMLException,
    create_error_from_exception,
    create_error_response,
    UnifiedErrorResponse,
)
from aml_fraud_detector.constants import ErrorCode
from aml_fraud_detector.entity import UnifiedPredictionResponse
from aml_fraud_detector.config import get_config_service, ConfigService
from aml_fraud_detector.logger import logging


_config_service: ConfigService = get_config_service()
_server_cfg = _config_service.config.server

application = Flask(__name__)
app = application

app.config["MAX_CONTENT_LENGTH"] = _server_cfg.max_content_length
if _server_cfg.enable_cors:
    CORS(
        app,
        resources={r"/*": {"origins": _server_cfg.cors_origins}},
        supports_credentials=True,
    )

predict_pipeline = PredictionPipeline(config_service=_config_service)

try:
    _effective_cfg_path = _config_service.export_effective_config(
        sanitized=True,
        include_metadata=True,
    )
    logging.info(f"Flask startup: effective config saved to {_effective_cfg_path}")
except Exception as _cfg_err:
    logging.warning(f"Flask startup: export effective config skipped: {_cfg_err}")

logging.info(
    f"Flask app initialized: host={_server_cfg.flask_host}, "
    f"port={_server_cfg.flask_port}, debug={_server_cfg.flask_debug}, "
    f"cors={_server_cfg.enable_cors}"
)
_config_service.print_summary()


def _get_trace_id() -> str:
    return str(uuid.uuid4())


def _parse_form_value(key: str, default=None):
    value = request.form.get(key, default)
    if value is None or value == "":
        return default
    return value


@app.route("/health", methods=["GET"])
def health_check():
    trace_id = _get_trace_id()
    return make_response(
        jsonify({
            "success": True,
            "status": "healthy",
            "trace_id": trace_id,
            "service": _config_service.config.app_name,
            "version": _config_service.config.app_version,
            "env": _config_service.config.env,
            "model_loaded": predict_pipeline._model is not None,
        }),
        200,
    )


@app.route("/api/v1/config", methods=["GET"])
def get_config_info():
    trace_id = _get_trace_id()
    try:
        sanitized_cfg = _config_service.get_effective_config(sanitized=True)
        meta = _config_service.get_config_metadata()
        return make_response(
            jsonify({
                "success": True,
                "trace_id": trace_id,
                "sanitized": True,
                "config": sanitized_cfg,
                "metadata": {
                    "profile": meta["profile"],
                    "env": meta["profile"],
                    "loaded_at": meta["loaded_at"],
                    "config_path": meta["config_path"],
                    "config_path_source": meta["config_path_source"],
                    "yaml_sources_count": len(meta["yaml_sources"]),
                    "env_override_count": meta["env_override_count"],
                },
            }),
            200,
        )
    except Exception as e:
        err_resp = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
        return make_response(jsonify(err_resp.to_dict()), err_resp.http_status)


@app.route("/config/export", methods=["POST"])
def export_config():
    trace_id = _get_trace_id()
    try:
        sanitized = request.args.get("sanitized", "true").lower() in ("1", "true", "yes", "on")
        output_path = _config_service.export_effective_config(
            sanitized=sanitized,
            include_metadata=True,
        )
        return make_response(
            jsonify({
                "success": True,
                "trace_id": trace_id,
                "sanitized": sanitized,
                "output_path": output_path,
            }),
            200,
        )
    except Exception as e:
        err_resp = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
        return make_response(jsonify(err_resp.to_dict()), err_resp.http_status)


@app.errorhandler(404)
def not_found(error):
    trace_id = _get_trace_id()
    error_resp = create_error_response(
        ErrorCode.INTERNAL_UNEXPECTED,
        trace_id=trace_id,
        detail="Endpoint not found",
    )
    return make_response(jsonify(error_resp.to_dict()), 404)


@app.errorhandler(405)
def method_not_allowed(error):
    trace_id = _get_trace_id()
    error_resp = create_error_response(
        ErrorCode.INTERNAL_UNEXPECTED,
        trace_id=trace_id,
        detail="Method not allowed",
    )
    return make_response(jsonify(error_resp.to_dict()), 405)


@app.errorhandler(413)
def payload_too_large(error):
    trace_id = _get_trace_id()
    error_resp = create_error_response(
        ErrorCode.INTERNAL_UNEXPECTED,
        trace_id=trace_id,
        detail=f"Request payload exceeds {_server_cfg.max_content_length} bytes",
    )
    return make_response(jsonify(error_resp.to_dict()), 413)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_data():
    trace_id = _get_trace_id()
    if request.method == "GET":
        return render_template("home.html")

    try:
        form_data = {
            "from_bank": _parse_form_value("from_bank"),
            "account": _parse_form_value("account"),
            "to_bank": _parse_form_value("to_bank"),
            "account_1": _parse_form_value("account_1"),
            "amount_received": _parse_form_value("amount_received"),
            "receiving_currency": _parse_form_value("receiving_currency"),
            "payment_currency": _parse_form_value("payment_currency"),
            "payment_format": _parse_form_value("payment_format"),
            "day": _parse_form_value("day"),
        }
        required = ["from_bank", "account", "to_bank", "account_1",
                    "amount_received", "receiving_currency",
                    "payment_currency", "payment_format", "day"]
        missing = [k for k in required if form_data.get(k) is None]
        if missing:
            err_resp = create_error_response(
                ErrorCode.INPUT_MISSING_FIELD,
                trace_id=trace_id,
                field=", ".join(missing),
            )
            return make_response(
                render_template("home.html", error=err_resp.to_user_display()),
                400,
            )

        custom_data = CustomData(
            from_bank=form_data["from_bank"],
            account=form_data["account"],
            to_bank=form_data["to_bank"],
            account_1=form_data["account_1"],
            amount_received=form_data["amount_received"],
            receiving_currency=form_data["receiving_currency"],
            payment_currency=form_data["payment_currency"],
            payment_format=form_data["payment_format"],
            day=form_data["day"],
        )
        validation = custom_data.validate()
        if not validation.is_valid:
            err_resp = create_error_response(
                ErrorCode.INPUT_INVALID_FORMAT,
                trace_id=trace_id,
                detail="; ".join(validation.errors),
            )
            return make_response(
                render_template("home.html", error={
                    "success": False,
                    "error_category": "input_validation",
                    "error_code": "E2003",
                    "message": "; ".join(validation.errors),
                    "suggestion": "请检查输入字段是否符合要求",
                }),
                400,
            )

        data = custom_data.get_data_as_DataFrame()
        prediction = predict_pipeline.predict(data)

        if not prediction.is_success():
            return make_response(
                render_template("home.html", error={
                    "success": False,
                    "error_category": prediction.error_detail.error_category.value,
                    "error_code": prediction.error_detail.error_code.value,
                    "message": prediction.error_detail.message,
                    "suggestion": "请稍后重试",
                }),
                500,
            )

        final_result = prediction.prediction
        fraud_prob = prediction.fraud_probability
        legit_prob = prediction.legit_probability
        risk_level = prediction.risk_explanation.risk_level.value if prediction.risk_explanation else "LOW"

        return render_template(
            "home.html",
            final_result=final_result,
            fraud_prob=f"{fraud_prob:.4f}",
            legit_prob=f"{legit_prob:.4f}",
            risk_level=risk_level,
            class_label=prediction.class_label,
        )

    except AMLException as e:
        err_resp = e.to_error_response(trace_id=trace_id)
        logging.error(f"Prediction endpoint failed (AMLException): {e}")
        return make_response(
            render_template("home.html", error=err_resp.to_user_display()),
            err_resp.http_status,
        )
    except Exception as e:
        err_resp = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
        logging.error(f"Prediction endpoint failed (unexpected): {e}", exc_info=True)
        return make_response(
            render_template("home.html", error=err_resp.to_user_display()),
            err_resp.http_status,
        )


@app.route("/api/v1/predict", methods=["POST"])
def api_predict():
    trace_id = _get_trace_id()
    try:
        payload = request.get_json(force=True, silent=True) or {}
        if not isinstance(payload, dict):
            err_resp = create_error_response(
                ErrorCode.INPUT_INVALID_FORMAT,
                trace_id=trace_id,
                detail="Request body must be a JSON object",
            )
            return make_response(jsonify(err_resp.to_dict()), 400)

        required_fields = ["from_bank", "account", "to_bank", "account_1",
                           "amount_received", "receiving_currency",
                           "payment_currency", "payment_format", "day"]
        missing = [f for f in required_fields if f not in payload]
        if missing:
            err_resp = create_error_response(
                ErrorCode.INPUT_MISSING_FIELD,
                trace_id=trace_id,
                field=", ".join(missing),
            )
            return make_response(jsonify(err_resp.to_dict()), 400)

        custom_data = CustomData(
            from_bank=payload["from_bank"],
            account=payload["account"],
            to_bank=payload["to_bank"],
            account_1=payload["account_1"],
            amount_received=payload["amount_received"],
            receiving_currency=payload["receiving_currency"],
            payment_currency=payload["payment_currency"],
            payment_format=payload["payment_format"],
            day=payload["day"],
        )
        validation = custom_data.validate()
        if not validation.is_valid:
            err_resp = create_error_response(
                ErrorCode.INPUT_INVALID_FORMAT,
                trace_id=trace_id,
                detail="; ".join(validation.errors),
            )
            return make_response(jsonify(err_resp.to_dict()), 400)

        data = custom_data.get_data_as_DataFrame()
        prediction = predict_pipeline.predict(data)
        unified = UnifiedPredictionResponse(
            success=prediction.is_success(),
            trace_id=trace_id,
            prediction=prediction,
            batch=None,
            error=prediction.error_detail if not prediction.is_success() else None,
        )
        http_status = 200 if prediction.is_success() else 500
        return make_response(jsonify(unified.to_dict()), http_status)

    except AMLException as e:
        err_resp = e.to_error_response(trace_id=trace_id)
        logging.error(f"API predict failed (AMLException): {e}")
        return make_response(jsonify(err_resp.to_dict()), err_resp.http_status)
    except Exception as e:
        err_resp = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
        logging.error(f"API predict failed (unexpected): {e}", exc_info=True)
        return make_response(jsonify(err_resp.to_dict()), err_resp.http_status)


@app.route("/api/v1/predict/batch", methods=["POST"])
def api_predict_batch():
    trace_id = _get_trace_id()
    try:
        payload = request.get_json(force=True, silent=True) or {}
        if not isinstance(payload, dict) or "transactions" not in payload:
            err_resp = create_error_response(
                ErrorCode.INPUT_INVALID_FORMAT,
                trace_id=trace_id,
                detail="Request body must contain 'transactions' array",
            )
            return make_response(jsonify(err_resp.to_dict()), 400)

        transactions = payload["transactions"]
        if not isinstance(transactions, list) or len(transactions) == 0:
            err_resp = create_error_response(
                ErrorCode.INPUT_INVALID_FORMAT,
                trace_id=trace_id,
                detail="'transactions' must be a non-empty array",
            )
            return make_response(jsonify(err_resp.to_dict()), 400)

        required_fields = ["from_bank", "account", "to_bank", "account_1",
                           "amount_received", "receiving_currency",
                           "payment_currency", "payment_format", "day"]

        records = []
        errors = []
        for idx, tx in enumerate(transactions):
            if not isinstance(tx, dict):
                errors.append(f"Transaction[{idx}]: must be an object")
                continue
            missing = [f for f in required_fields if f not in tx]
            if missing:
                errors.append(f"Transaction[{idx}]: missing fields {missing}")
                continue
            try:
                custom_data = CustomData(
                    from_bank=tx["from_bank"],
                    account=tx["account"],
                    to_bank=tx["to_bank"],
                    account_1=tx["account_1"],
                    amount_received=tx["amount_received"],
                    receiving_currency=tx["receiving_currency"],
                    payment_currency=tx["payment_currency"],
                    payment_format=tx["payment_format"],
                    day=tx["day"],
                )
                validation = custom_data.validate()
                if not validation.is_valid:
                    errors.append(f"Transaction[{idx}]: {'; '.join(validation.errors)}")
                    continue
                df_row = custom_data.get_data_as_DataFrame()
                records.append(df_row)
            except Exception as row_err:
                errors.append(f"Transaction[{idx}]: {row_err}")

        if errors and not records:
            err_resp = create_error_response(
                ErrorCode.INPUT_INVALID_FORMAT,
                trace_id=trace_id,
                detail="All transactions failed validation: " + "; ".join(errors),
            )
            return make_response(jsonify(err_resp.to_dict()), 400)

        import pandas as pd
        data_df = pd.concat(records, ignore_index=True)
        batch_result = predict_pipeline.predict_batch(data_df)

        if errors:
            for i, pred in enumerate(batch_result.predictions):
                if pred.transaction_id and pred.transaction_id.isdigit():
                    pass

        unified = UnifiedPredictionResponse(
            success=batch_result.is_success(),
            trace_id=trace_id,
            prediction=None,
            batch=batch_result,
            error=batch_result.error_detail if not batch_result.is_success() else None,
        )
        resp = unified.to_dict()
        if errors:
            resp["warnings"] = {
                "validation_errors": errors,
                "processed_count": batch_result.total_count,
                "submitted_count": len(transactions),
            }
        http_status = 200 if batch_result.is_success() else 500
        return make_response(jsonify(resp), http_status)

    except AMLException as e:
        err_resp = e.to_error_response(trace_id=trace_id)
        logging.error(f"API batch predict failed (AMLException): {e}")
        return make_response(jsonify(err_resp.to_dict()), err_resp.http_status)
    except Exception as e:
        err_resp = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
        logging.error(f"API batch predict failed (unexpected): {e}", exc_info=True)
        return make_response(jsonify(err_resp.to_dict()), err_resp.http_status)


@app.route("/api/v1/model/info", methods=["GET"])
def api_model_info():
    trace_id = _get_trace_id()
    try:
        version_info = predict_pipeline.get_model_version_info()
        model_meta = None
        if predict_pipeline._model_metadata:
            model_meta = dict(predict_pipeline._model_metadata)
        return make_response(
            jsonify({
                "success": True,
                "trace_id": trace_id,
                "model_version_info": {
                    "model_version": version_info.model_version,
                    "model_name": version_info.model_name,
                    "training_time": version_info.training_time,
                    "selection_metric": version_info.selection_metric,
                    "best_metric_value": version_info.best_metric_value,
                    "feature_schema_version": version_info.feature_schema_version,
                    "artifact_path": version_info.artifact_path,
                },
                "prediction_config": {
                    "artifacts_dir": predict_pipeline.artifacts_dir,
                    "model_path": predict_pipeline.model_path,
                    "preprocessor_path": predict_pipeline.preprocessor_path,
                    "feature_metadata_path": predict_pipeline.feature_metadata_path,
                    "model_metadata_path": predict_pipeline.model_metadata_path,
                    "risk_thresholds": predict_pipeline.risk_thresholds,
                    "lazy_load": predict_pipeline.enable_lazy_load,
                },
                "model_metadata_raw": model_meta,
            }),
            200,
        )
    except Exception as e:
        err_resp = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
        return make_response(jsonify(err_resp.to_dict()), err_resp.http_status)


def run_server():
    cfg = _config_service.config.server
    host = cfg.flask_host
    port = int(cfg.flask_port)
    debug = bool(cfg.flask_debug)
    logging.info(f"Starting Flask server on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server()

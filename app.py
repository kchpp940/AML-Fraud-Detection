import sys
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
from aml_fraud_detector.config import get_config_service
from aml_fraud_detector.logger import logging

application = Flask(__name__)
app = application
CORS(app)

_config_service = get_config_service()
predict_pipeline = PredictionPipeline(config_service=_config_service)


def _get_trace_id() -> str:
    return str(uuid.uuid4())


def _parse_form_value(key: str, default=None):
    value = request.form.get(key, default)
    if value is None or value == "":
        return default
    return value


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


def run_server():
    cfg = _config_service.config.server
    host = cfg.flask_host
    port = int(cfg.flask_port)
    debug = bool(cfg.flask_debug)
    logging.info(f"Starting Flask server on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server()

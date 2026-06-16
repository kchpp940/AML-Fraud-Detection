import sys
import uuid
from flask import Flask, request, render_template, jsonify, make_response
from flask_cors import CORS

from aml_fraud_detector.bootstrap import ApplicationContainer
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.exception import (
    AMLException,
    create_error_from_exception,
    create_error_response,
    UnifiedErrorResponse,
)
from aml_fraud_detector.constants import ErrorCode
from aml_fraud_detector.entity import UnifiedPredictionResponse, ProcessStatus
from aml_fraud_detector.logger import logging

application = Flask(__name__)
app = application
CORS(app)

container = ApplicationContainer()


def _get_trace_id() -> str:
    return str(uuid.uuid4())


def _parse_form_value(key: str, default=None):
    value = request.form.get(key, default)
    if value is None or value == "":
        return default
    return value


def _get_pipeline():
    if container.prediction_pipeline is not None:
        return container.prediction_pipeline
    return PredictionPipeline()


@app.route("/health", methods=["GET"])
def health():
    trace_id = _get_trace_id()
    try:
        health_data = container.get_health_dict()
        status_code = 200 if health_data["overall"] == "healthy" else 503
        response = {
            "success": health_data["overall"] == "healthy",
            "trace_id": trace_id,
            **health_data,
        }
        return make_response(jsonify(response), status_code)
    except Exception as e:
        error_resp = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
        return make_response(jsonify(error_resp.to_dict()), 500)


@app.route("/api/version", methods=["GET"])
def api_version():
    trace_id = _get_trace_id()
    try:
        version_data = container.get_version_dict()
        response = {
            "success": True,
            "trace_id": trace_id,
            **version_data,
        }
        return make_response(jsonify(response), 200)
    except Exception as e:
        error_resp = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
        return make_response(jsonify(error_resp.to_dict()), 500)


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
def predict():
    trace_id = _get_trace_id()
    try:
        if request.method == "GET":
            return render_template("home.html")

        from_bank = _parse_form_value("from_bank")
        account = _parse_form_value("account")
        to_bank = _parse_form_value("to_bank")
        account_1 = _parse_form_value("account_1")
        amount_received = _parse_form_value("amount_received")
        receiving_currency = _parse_form_value("receiving_currency")
        payment_currency = _parse_form_value("payment_currency")
        payment_format = _parse_form_value("payment_format")
        day = _parse_form_value("day")

        custom_data = CustomData(
            from_bank=from_bank,
            account=account,
            to_bank=to_bank,
            account_1=account_1,
            amount_received=amount_received,
            receiving_currency=receiving_currency,
            payment_currency=payment_currency,
            payment_format=payment_format,
            day=day,
        )

        validation = custom_data.validate()
        if not validation.is_valid:
            error_resp = create_error_response(
                ErrorCode.INPUT_INVALID_FORMAT,
                trace_id=trace_id,
                detail="; ".join(validation.errors),
            )
            return make_response(jsonify(error_resp.to_dict()), 400)

        data_df = custom_data.get_data_as_DataFrame()

        predict_pipeline = _get_pipeline()
        result = predict_pipeline.predict(data_df)

        if not result.is_success():
            error_resp = create_error_response(
                ErrorCode.PREDICTION_FAILED,
                trace_id=trace_id,
                detail=result.error_reason or "Prediction failed",
            )
            return make_response(jsonify(error_resp.to_dict()), 500)

        response = UnifiedPredictionResponse(
            model_version=container.get_model_version(),
            validation=validation,
            single_prediction=result,
        )

        return render_template(
            "home.html",
            results=result.to_dict(),
            prediction=result.class_label,
        )

    except AMLException as e:
        error_resp = e.to_error_response(trace_id=trace_id)
        return make_response(jsonify(error_resp.to_dict()), error_resp.http_status)
    except Exception as e:
        error_resp = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
        return make_response(jsonify(error_resp.to_dict()), 500)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    trace_id = _get_trace_id()
    try:
        req_data = request.get_json(force=True)

        custom_data = CustomData(
            from_bank=req_data.get("from_bank"),
            account=req_data.get("account"),
            to_bank=req_data.get("to_bank"),
            account_1=req_data.get("account_1"),
            amount_received=req_data.get("amount_received"),
            receiving_currency=req_data.get("receiving_currency"),
            payment_currency=req_data.get("payment_currency"),
            payment_format=req_data.get("payment_format"),
            day=req_data.get("day"),
        )

        validation = custom_data.validate()
        if not validation.is_valid:
            response = UnifiedPredictionResponse(
                model_version=container.get_model_version(),
                validation=validation,
                error=create_error_response(
                    ErrorCode.INPUT_INVALID_FORMAT,
                    trace_id=trace_id,
                    detail="; ".join(validation.errors),
                ),
            )
            return make_response(jsonify(response.to_dict()), 400)

        data_df = custom_data.get_data_as_DataFrame()

        predict_pipeline = _get_pipeline()
        result = predict_pipeline.predict(data_df)

        response = UnifiedPredictionResponse(
            model_version=container.get_model_version(),
            validation=validation,
            single_prediction=result,
        )

        if not result.is_success():
            response.error = create_error_response(
                ErrorCode.PREDICTION_FAILED,
                trace_id=trace_id,
                detail=result.error_reason or "Prediction failed",
            )
            return make_response(jsonify(response.to_dict()), 500)

        return make_response(jsonify(response.to_dict()), 200)

    except AMLException as e:
        error_resp = e.to_error_response(trace_id=trace_id)
        return make_response(jsonify(error_resp.to_dict()), error_resp.http_status)
    except Exception as e:
        error_resp = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
        return make_response(jsonify(error_resp.to_dict()), 500)


if __name__ == "__main__":
    if not container.health.is_healthy():
        logging.error(
            f"Application failed to start. Health status: {container.health.overall.value}"
        )
        for comp in container.health.components:
            if comp.status == "error":
                logging.error(f"  - {comp.name}: {comp.message}")
        sys.exit(1)

    logging.info("Starting Flask application")
    app.run(host="0.0.0.0", port=8080, debug=False)

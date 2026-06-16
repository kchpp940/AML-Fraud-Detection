import sys
import uuid
from flask import Flask, request, render_template, jsonify, make_response

from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.exception import (
    AMLException,
    InputValidationException,
    create_error_from_exception,
    create_error_response,
    UnifiedErrorResponse,
)
from aml_fraud_detector.constants import ErrorCode, HTTP_STATUS_CODES, ErrorCategory
from aml_fraud_detector.entity import (
    UnifiedPredictionResponse,
    ValidationStatus,
    ProcessStatus,
)
from aml_fraud_detector.logger import logging

application = Flask(__name__)
app = application

predict_pipeline = PredictionPipeline()


def _get_trace_id() -> str:
    return str(uuid.uuid4())


def _parse_form_value(key: str, default=None):
    value = request.form.get(key, default)
    if value is None or value == "":
        return default
    return value


def _handle_aml_exception(e: AMLException, trace_id: str):
    error_resp = e.to_error_response(trace_id=trace_id)
    return make_response(jsonify(error_resp.to_dict()), error_resp.http_status)


def _handle_unexpected_exception(e: Exception, trace_id: str):
    error_resp = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
    return make_response(jsonify(error_resp.to_dict()), error_resp.http_status)


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


@app.errorhandler(AMLException)
def aml_exception_handler(e):
    trace_id = _get_trace_id()
    logging.error(f"AMLException caught by Flask error handler: {e}")
    return _handle_aml_exception(e, trace_id)


@app.errorhandler(Exception)
def generic_exception_handler(e):
    trace_id = _get_trace_id()
    logging.error(f"Unexpected exception caught by Flask error handler: {e}", exc_info=True)
    return _handle_unexpected_exception(e, trace_id)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    trace_id = _get_trace_id()

    if request.method == "GET":
        return render_template("home.html", results=None, error=None)

    try:
        from_bank = _parse_form_value("from_bank")
        account = _parse_form_value("account")
        to_bank = _parse_form_value("to_bank")
        account_1 = _parse_form_value("account_1")
        amount_received = _parse_form_value("amount_received")
        receiving_currency = _parse_form_value("receiving_currency")
        payment_currency = _parse_form_value("payment_currency")
        payment_format = _parse_form_value("payment_format")

        if from_bank is None:
            raise InputValidationException(
                ErrorCode.INPUT_MISSING_FIELD,
                error_details=sys,
                field="from_bank",
            )
        if to_bank is None:
            raise InputValidationException(
                ErrorCode.INPUT_MISSING_FIELD,
                error_details=sys,
                field="to_bank",
            )
        if amount_received is None:
            raise InputValidationException(
                ErrorCode.INPUT_MISSING_FIELD,
                error_details=sys,
                field="amount_received",
            )

        data = CustomData(
            from_bank=from_bank,
            account=account or "",
            to_bank=to_bank,
            account_1=account_1 or "",
            amount_received=amount_received,
            receiving_currency=receiving_currency or "",
            payment_currency=payment_currency or "",
            payment_format=payment_format or "",
            day="Monday",
        )

        validation = data.validate()
        if not validation.is_valid:
            error_resp = create_error_response(
                ErrorCode.INPUT_INVALID_FORMAT,
                trace_id=trace_id,
                field="multiple",
                value="; ".join(validation.errors),
            )
            return render_template(
                "home.html",
                results=None,
                error=error_resp.to_user_display(),
            )

        pred_df = data.get_data_as_DataFrame()
        result = predict_pipeline.predict(pred_df)

        if not result.is_success():
            error_resp = create_error_response(
                result.error_detail.error_code,
                trace_id=trace_id,
                **result.error_detail.context,
            )
            return render_template(
                "home.html",
                results=None,
                error=error_resp.to_user_display(),
            )

        display_result = {
            "prediction": result.class_label,
            "fraud_probability": f"{result.fraud_probability:.4f}",
            "risk_level": result.risk_level.value,
        }
        return render_template("home.html", results=display_result, error=None)

    except AMLException as e:
        logging.error(f"Prediction page error: {e}")
        error_resp = e.to_error_response(trace_id=trace_id)
        return render_template(
            "home.html",
            results=None,
            error=error_resp.to_user_display(),
        )
    except Exception as e:
        logging.error("Unexpected error in predict_datapoint", exc_info=True)
        error_resp = create_error_from_exception(e, trace_id=trace_id, error_details=sys)
        return render_template(
            "home.html",
            results=None,
            error=error_resp.to_user_display(),
        )


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
        missing = [f for f in required_fields if f not in req_data or req_data[f] is None]
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
        result = predict_pipeline.predict(pred_df)

        version_info = predict_pipeline.get_model_version_info()
        response = UnifiedPredictionResponse(
            model_version=version_info,
            validation=validation,
            single_prediction=result,
        )

        if not result.is_success():
            error_resp = create_error_response(
                result.error_detail.error_code,
                trace_id=trace_id,
                **result.error_detail.context,
            )
            response.error = error_resp
            return make_response(jsonify(response.to_dict()), error_resp.http_status)

        return make_response(jsonify(response.to_dict()), 200)

    except AMLException as e:
        logging.error(f"API predict error: {e}")
        return _handle_aml_exception(e, trace_id)
    except Exception as e:
        logging.error("Unexpected error in api_predict", exc_info=True)
        return _handle_unexpected_exception(e, trace_id)


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

        import pandas as pd
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

        for col in ["from_bank", "to_bank"]:
            if col in df.columns:
                df[col] = df[col].astype("object")

        result = predict_pipeline.predict_batch(df)
        version_info = predict_pipeline.get_model_version_info()
        response = UnifiedPredictionResponse(
            model_version=version_info,
            batch_prediction=result,
        )

        if not result.is_success():
            error_resp = create_error_response(
                result.error_detail.error_code,
                trace_id=trace_id,
                **result.error_detail.context,
            )
            response.error = error_resp
            return make_response(jsonify(response.to_dict()), error_resp.http_status)

        return make_response(jsonify(response.to_dict()), 200)

    except AMLException as e:
        logging.error(f"API batch predict error: {e}")
        return _handle_aml_exception(e, trace_id)
    except Exception as e:
        logging.error("Unexpected error in api_predict_batch", exc_info=True)
        return _handle_unexpected_exception(e, trace_id)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

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
from aml_fraud_detector.logger import logging

application = Flask(__name__)
app = application
CORS(app)

predict_pipeline = PredictionPipeline()


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
    error_resp = create_error_response
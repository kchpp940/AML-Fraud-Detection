from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd

from aml_fraud_detector.artifact_validator import ArtifactValidator
from aml_fraud_detector.constants import ErrorCode
from aml_fraud_detector.entity import (
    REQUIRED_INPUT_FIELDS,
    BatchPredictionResult,
    ModelVersionInfo,
    PredictionResult,
    TrainingPipelineResult,
    UnifiedPredictionResponse,
    ValidationStatus,
)
from aml_fraud_detector.exception import (
    AMLException,
    DataQualityException,
    InputValidationException,
    MetadataValidationException,
    wrap_exception,
)
from aml_fraud_detector.logger import logging as _pkg_logger
from aml_fraud_detector.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from aml_fraud_detector.pipeline.training_pipeline import run_training_pipeline
from aml_fraud_detector.presentation.display_builders import batch_predictions_to_dataframe


def _trace_id() -> str:
    return uuid.uuid4().hex[:16]


def _print_dict(data: Dict[str, Any]) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _print_error(exc: AMLException, trace_id: str) -> None:
    _print_dict(exc.to_error_response(trace_id=trace_id).to_dict())


def _load_context(artifacts_dir: str):
    validator = ArtifactValidator(artifacts_dir=artifacts_dir)
    pipeline = PredictionPipeline(artifacts_dir=artifacts_dir)
    mvi = validator.load_model_version_info() or ModelVersionInfo()
    vs = validator.validate_artifacts(verify_digests=False)
    return validator, pipeline, mvi, vs


def _ensure_artifacts_valid(vs: ValidationStatus, tid: str, mvi: ModelVersionInfo):
    if not vs.is_valid:
        exc = MetadataValidationException(
            ErrorCode.ARTIFACT_MISSING,
            error_details=sys,
            artifact="; ".join(vs.errors),
        )
        _print_error(exc, tid)
        sys.exit(1)


def cmd_train(args: argparse.Namespace) -> int:
    _pkg_logger.info("CLI: train")
    tid = _trace_id()
    try:
        result: TrainingPipelineResult = run_training_pipeline(config_path=args.config)
        if result.is_success():
            _print_dict({
                "success": True,
                "status": "success",
                "summary": result.summary or {},
                "artifacts": result.artifacts or {},
                "trace_id": tid,
            })
            return 0
        if result.error_detail:
            from aml_fraud_detector.exception import UnifiedErrorResponse
            err_resp = UnifiedErrorResponse(
                success=False,
                status="error",
                error=result.error_detail,
                http_status=500,
                trace_id=tid,
            )
            _print_dict(err_resp.to_dict())
        else:
            _print_dict({
                "success": False,
                "status": "error",
                "error_reason": result.error_reason or "训练失败",
                "trace_id": tid,
            })
        return 1
    except AMLException as e:
        _print_error(e, tid)
        return 1
    except Exception as e:
        wrapped = wrap_exception(e, error_details=sys)
        _print_error(wrapped, tid)
        return 1


def cmd_validate_artifacts(args: argparse.Namespace) -> int:
    _pkg_logger.info("CLI: validate-artifacts")
    tid = _trace_id()
    artifacts_dir = args.artifacts_dir or "artifacts"
    try:
        validator = ArtifactValidator(artifacts_dir=artifacts_dir)
        vs = validator.validate_artifacts(verify_digests=args.verify_digests)
        mvi = validator.load_model_version_info() or ModelVersionInfo()
        if not vs.is_valid:
            exc = MetadataValidationException(
                ErrorCode.ARTIFACT_MISSING,
                error_details=sys,
                artifact="; ".join(vs.errors),
            )
            _print_error(exc, tid)
            return 1
        resp = UnifiedPredictionResponse(
            model_version=mvi,
            validation=vs,
        )
        _print_dict(resp.to_dict())
        return 0
    except AMLException as e:
        _print_error(e, tid)
        return 1
    except Exception as e:
        wrapped = wrap_exception(e, error_details=sys)
        _print_error(wrapped, tid)
        return 1


def cmd_show_version(args: argparse.Namespace) -> int:
    _pkg_logger.info("CLI: show-version")
    tid = _trace_id()
    artifacts_dir = args.artifacts_dir or "artifacts"
    try:
        validator = ArtifactValidator(artifacts_dir=artifacts_dir)
        mvi = validator.load_model_version_info()
        vs = validator.validate_artifacts(verify_digests=False)
        if mvi is None:
            exc = MetadataValidationException(
                ErrorCode.METADATA_FILE_NOT_FOUND,
                error_details=sys,
                path=os.path.join(artifacts_dir, "model_metadata.json"),
            )
            _print_error(exc, tid)
            return 1
        resp = UnifiedPredictionResponse(
            model_version=mvi,
            validation=vs,
        )
        _print_dict(resp.to_dict())
        return 0
    except AMLException as e:
        _print_error(e, tid)
        return 1
    except Exception as e:
        wrapped = wrap_exception(e, error_details=sys)
        _print_error(wrapped, tid)
        return 1


def _collect_input_data(args: argparse.Namespace) -> Dict[str, Any]:
    if args.input_json:
        try:
            data = json.loads(args.input_json)
        except json.JSONDecodeError as e:
            raise InputValidationException(
                ErrorCode.INPUT_INVALID_FORMAT,
                error_details=sys,
                field="input_json",
                value=str(e),
            )
    else:
        data = {}
        arg_map = {
            "from_bank": args.from_bank,
            "account": args.account,
            "to_bank": args.to_bank,
            "account_1": args.account_1,
            "amount_received": args.amount_received,
            "receiving_currency": args.receiving_currency,
            "payment_currency": args.payment_currency,
            "payment_format": args.payment_format,
            "day": args.day,
        }
        for f in REQUIRED_INPUT_FIELDS:
            if f not in arg_map or arg_map[f] is None:
                raise InputValidationException(
                    ErrorCode.INPUT_MISSING_FIELD,
                    error_details=sys,
                    field=f,
                )
            data[f] = arg_map[f]
    return data


def cmd_predict_one(args: argparse.Namespace) -> int:
    _pkg_logger.info("CLI: predict-one")
    tid = _trace_id()
    artifacts_dir = args.artifacts_dir or "artifacts"
    try:
        validator, pipeline, mvi, vs = _load_context(artifacts_dir)
        _ensure_artifacts_valid(vs, tid, mvi)
        data = _collect_input_data(args)
        cd = CustomData(**{k: data[k] for k in REQUIRED_INPUT_FIELDS})
        if hasattr(cd, "validate"):
            v = cd.validate()
            if not v.is_valid:
                raise InputValidationException(
                    ErrorCode.INPUT_INVALID_FORMAT,
                    error_details=sys,
                    field="validation",
                    value="; ".join(v.errors),
                )
        df = cd.get_data_as_DataFrame()
        result: PredictionResult = pipeline.predict(df)
        resp = UnifiedPredictionResponse(
            model_version=mvi,
            validation=vs,
            single_prediction=result,
        )
        _print_dict(resp.to_dict())
        return 0 if result.is_success() else 1
    except AMLException as e:
        _print_error(e, tid)
        return 1
    except Exception as e:
        wrapped = wrap_exception(e, error_details=sys)
        _print_error(wrapped, tid)
        return 1


def cmd_predict_batch(args: argparse.Namespace) -> int:
    _pkg_logger.info("CLI: predict-batch")
    tid = _trace_id()
    artifacts_dir = args.artifacts_dir or "artifacts"
    try:
        validator, pipeline, mvi, vs = _load_context(artifacts_dir)
        _ensure_artifacts_valid(vs, tid, mvi)
        input_path = args.input_file
        if not os.path.isfile(input_path):
            raise DataQualityException(
                ErrorCode.DATA_SOURCE_NOT_FOUND,
                error_details=sys,
                path=input_path,
            )
        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            raise DataQualityException(
                ErrorCode.DATA_CORRUPTED,
                error_details=sys,
                path=input_path,
                detail=str(e),
            )
        if len(df) == 0:
            raise DataQualityException(
                ErrorCode.DATA_EMPTY,
                error_details=sys,
                rows=0,
            )
        result: BatchPredictionResult = pipeline.predict_batch(df)
        if args.output_format in ("json", "both"):
            resp = UnifiedPredictionResponse(
                model_version=mvi,
                validation=vs,
                batch_prediction=result,
            )
            _print_dict(resp.to_dict())
        if result.is_success() and args.output_format in ("csv", "both"):
            batch_df = batch_predictions_to_dataframe(result)
            if args.output_file:
                batch_df.to_csv(args.output_file, index=False)
                if args.output_format == "both":
                    print()
                print(f"Batch predictions saved to: {args.output_file}")
            else:
                print(batch_df.to_csv(index=False))
        return 0 if result.is_success() else 1
    except AMLException as e:
        _print_error(e, tid)
        return 1
    except Exception as e:
        wrapped = wrap_exception(e, error_details=sys)
        _print_error(wrapped, tid)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aml-fraud",
        description="AML Fraud Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train
  aml-fraud train --config config/training_config.yaml

  # Validate artifacts
  aml-fraud validate-artifacts --artifacts-dir artifacts --verify-digests

  # Show model version
  aml-fraud show-version --artifacts-dir artifacts

  # Single prediction (named params)
  aml-fraud predict-one --from-bank 1 --account ACC001 --to-bank 2 \\
    --account-1 ACC002 --amount-received 1000.0 --receiving-currency USD \\
    --payment-currency USD --payment-format ACH --day Monday

  # Single prediction (JSON)
  aml-fraud predict-one --input-json '{"from_bank":1,"account":"ACC001",...}'

  # Batch prediction
  aml-fraud predict-batch --input-file tx.csv --output-format csv --output-file out.csv
        """,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tp = sub.add_parser("train", help="Run training pipeline")
    tp.add_argument("--config", type=str, default=None, help="Training config path")
    tp.set_defaults(func=cmd_train)

    vp = sub.add_parser("validate-artifacts", help="Validate model artifacts")
    vp.add_argument("--artifacts-dir", type=str, default=None, help="Artifacts dir (default: artifacts)")
    vp.add_argument("--verify-digests", action="store_true", help="Verify SHA256 digests")
    vp.set_defaults(func=cmd_validate_artifacts)

    sp = sub.add_parser("show-version", help="Show model version info")
    sp.add_argument("--artifacts-dir", type=str, default=None, help="Artifacts dir (default: artifacts)")
    sp.set_defaults(func=cmd_show_version)

    pp = sub.add_parser("predict-one", help="Single prediction")
    pp.add_argument("--artifacts-dir", type=str, default=None, help="Artifacts dir (default: artifacts)")
    pp.add_argument("--input-json", type=str, default=None, help="All input fields as JSON")
    pp.add_argument("--from-bank", type=int)
    pp.add_argument("--account", type=str)
    pp.add_argument("--to-bank", type=int)
    pp.add_argument("--account-1", type=str, dest="account_1")
    pp.add_argument("--amount-received", type=float)
    pp.add_argument("--receiving-currency", type=str)
    pp.add_argument("--payment-currency", type=str)
    pp.add_argument("--payment-format", type=str)
    pp.add_argument("--day", type=str)
    pp.set_defaults(func=cmd_predict_one)

    bp = sub.add_parser("predict-batch", help="Batch prediction from CSV")
    bp.add_argument("--artifacts-dir", type=str, default=None, help="Artifacts dir (default: artifacts)")
    bp.add_argument("--input-file", type=str, required=True, help="Input CSV path")
    bp.add_argument(
        "--output-format",
        type=str,
        default="json",
        choices=["json", "csv", "both"],
        help="Output format (default: json)",
    )
    bp.add_argument("--output-file", type=str, default=None, help="Output CSV path (for csv/both)")
    bp.set_defaults(func=cmd_predict_batch)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nCancelled")
        return 130
    except Exception as e:
        tid = _trace_id()
        wrapped = wrap_exception(e, error_details=sys)
        _print_error(wrapped, tid)
        return 1


if __name__ == "__main__":
    sys.exit(main())

import sys
import os
import re
import datetime
import calendar
import pandas as pd

from aml_fraud_detector.exception import CustomerException
from aml_fraud_detector.logger import logging
from aml_fraud_detector.utils.main_utils import load_object


class InputValidationError(Exception):
    pass


class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            predictions = model.predict(data_scaled)
            return predictions
        except Exception as e:
            raise CustomerException(e, sys)

    def predict_proba(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            predictions_prob = model.predict_proba(data_scaled)
            return predictions_prob
        except Exception as e:
            raise CustomerException(e, sys)


VALID_DAY_NAMES = {
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
    '星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日', '星期天',
    '周一', '周二', '周三', '周四', '周五', '周六', '周日'
}

DAY_NAME_TO_ENGLISH = {
    'monday': 'Monday', 'mon': 'Monday', '星期一': 'Monday', '周一': 'Monday',
    'tuesday': 'Tuesday', 'tue': 'Tuesday', '星期二': 'Tuesday', '周二': 'Tuesday',
    'wednesday': 'Wednesday', 'wed': 'Wednesday', '星期三': 'Wednesday', '周三': 'Wednesday',
    'thursday': 'Thursday', 'thu': 'Thursday', '星期四': 'Thursday', '周四': 'Thursday',
    'friday': 'Friday', 'fri': 'Friday', '星期五': 'Friday', '周五': 'Friday',
    'saturday': 'Saturday', 'sat': 'Saturday', '星期六': 'Saturday', '周六': 'Saturday',
    'sunday': 'Sunday', 'sun': 'Sunday', '星期日': 'Sunday', '星期天': 'Sunday', '周日': 'Sunday'
}


def _parse_amount(value):
    if value is None:
        raise InputValidationError("金额 (amount_received) 不能为空")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        raise InputValidationError("金额 (amount_received) 不能为空")
    s = s.replace(',', '').replace('，', '')
    s = re.sub(r'[^\d.\-]', '', s)
    if not s or s in ('-', '.', '-.'):
        raise InputValidationError(f"金额 (amount_received) 格式无效: '{value}'")
    try:
        return float(s)
    except ValueError:
        raise InputValidationError(f"金额 (amount_received) 格式无效: '{value}'")


def _parse_int(value, field_name):
    if value is None:
        raise InputValidationError(f"{field_name} 不能为空")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise InputValidationError(f"{field_name} 必须是整数，得到: '{value}'")
    s = str(value).strip()
    if not s:
        raise InputValidationError(f"{field_name} 不能为空")
    s = s.replace(',', '').replace('，', '')
    try:
        return int(s)
    except ValueError:
        raise InputValidationError(f"{field_name} 必须是整数，得到: '{value}'")


def _parse_str(value, field_name, allow_empty=False):
    if value is None:
        if allow_empty:
            return ""
        raise InputValidationError(f"{field_name} 不能为空")
    s = str(value).strip()
    if not s and not allow_empty:
        raise InputValidationError(f"{field_name} 不能为空")
    return s


def _normalize_day(value):
    if value is None:
        raise InputValidationError("日期/星期 (day) 不能为空")

    s = str(value).strip()
    if not s:
        raise InputValidationError("日期/星期 (day) 不能为空")

    if isinstance(value, datetime.date):
        return value.strftime("%A")

    if isinstance(value, datetime.datetime):
        return value.strftime("%A")

    dt = None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y", "%Y%m%d"):
        try:
            dt = datetime.datetime.strptime(s, fmt)
            break
        except ValueError:
            continue

    if dt is not None:
        return dt.strftime("%A")

    try:
        weekday_int = int(s)
        if 0 <= weekday_int <= 6:
            return calendar.day_name[weekday_int]
        if 1 <= weekday_int <= 7:
            return calendar.day_name[weekday_int - 1]
    except ValueError:
        pass

    lower_key = s.lower()
    if lower_key in DAY_NAME_TO_ENGLISH:
        return DAY_NAME_TO_ENGLISH[lower_key]

    raise InputValidationError(
        f"日期/星期 (day) 格式无效: '{value}'。"
        f"支持格式: 日期 (YYYY-MM-DD, YYYY/MM/DD), 星期数字 (0-6 或 1-7), "
        f"星期名称 (Monday, Monday, 星期一, 周一等)。"
    )


class CustomData:
    def __init__(self,
                 from_bank,
                 account,
                 to_bank,
                 account_1,
                 amount_received,
                 receiving_currency,
                 payment_currency,
                 payment_format,
                 day):
        self.errors = []
        self.from_bank = None
        self.account = None
        self.to_bank = None
        self.account_1 = None
        self.amount_received = None
        self.receiving_currency = None
        self.payment_currency = None
        self.payment_format = None
        self.day = None
        self._validate_and_transform(
            from_bank, account, to_bank, account_1,
            amount_received, receiving_currency, payment_currency,
            payment_format, day
        )

    def _validate_and_transform(self, from_bank, account, to_bank, account_1,
                                amount_received, receiving_currency, payment_currency,
                                payment_format, day):
        try:
            self.from_bank = _parse_int(from_bank, "发起银行编号 (from_bank)")
        except InputValidationError as e:
            self.errors.append(str(e))

        try:
            self.account = _parse_str(account, "发起账户 (account)")
        except InputValidationError as e:
            self.errors.append(str(e))

        try:
            self.to_bank = _parse_int(to_bank, "接收银行编号 (to_bank)")
        except InputValidationError as e:
            self.errors.append(str(e))

        try:
            self.account_1 = _parse_str(account_1, "接收账户 (account_1)")
        except InputValidationError as e:
            self.errors.append(str(e))

        try:
            self.amount_received = _parse_amount(amount_received)
        except InputValidationError as e:
            self.errors.append(str(e))

        try:
            self.receiving_currency = _parse_str(receiving_currency, "接收货币 (receiving_currency)")
        except InputValidationError as e:
            self.errors.append(str(e))

        try:
            self.payment_currency = _parse_str(payment_currency, "支付货币 (payment_currency)")
        except InputValidationError as e:
            self.errors.append(str(e))

        try:
            self.payment_format = _parse_str(payment_format, "支付方式 (payment_format)")
        except InputValidationError as e:
            self.errors.append(str(e))

        try:
            self.day = _normalize_day(day)
        except InputValidationError as e:
            self.errors.append(str(e))

        if self.errors:
            raise InputValidationError("；".join(self.errors))

    def get_data_as_DataFrame(self):
        try:
            custom_data_input_dict = {
                "from_bank": [str(self.from_bank)],
                "account": [self.account],
                "to_bank": [str(self.to_bank)],
                "account_1": [self.account_1],
                "amount_received": [self.amount_received],
                "receiving_currency": [self.receiving_currency],
                "payment_currency": [self.payment_currency],
                "payment_format": [self.payment_format],
                "day": [self.day]
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomerException(e, sys)

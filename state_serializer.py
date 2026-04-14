import math
import typing
import zlib
import base64
import json
import datetime

import pandas as pd

_MISSING = object()
_JSON_KEY_TYPES = (str, int, float, bool, type(None))

# https://chatgpt.com/c/69deba47-3538-8394-b305-2c4639d09a12
class DateTimePandasEncoder(json.JSONEncoder):
    """
    JSON encoder that serializes:
      - datetime.datetime
      - pandas.Timestamp

    Output format uses tagged objects so the decoder can restore the right type.
    """

    def default(self, obj: typing.Any) -> typing.Any:
        if isinstance(obj, pd.Timestamp):
            return {
                "__type__": "pandas.Timestamp",
                "value": obj.isoformat(),
            }

        if isinstance(obj, datetime.datetime):
            return {
                "__type__": "datetime.datetime",
                "value": obj.isoformat(),
            }

        return super().default(obj)


def datetime_pandas_object_hook(obj: dict[str, typing.Any]) -> typing.Any:
    """
    JSON decoder hook that deserializes tagged datetime / pandas.Timestamp objects.
    """
    type_name = obj.get("__type__")
    value = obj.get("value")

    if type_name == "datetime.datetime":
        if not isinstance(value, str):
            raise ValueError("Invalid value for datetime.datetime")
        return datetime.datetime.fromisoformat(value)

    if type_name == "pandas.Timestamp":
        if not isinstance(value, str):
            raise ValueError("Invalid value for pandas.Timestamp")
        return pd.Timestamp(value)

    return obj


class DateTimePandasDecoder(json.JSONDecoder):
    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        super().__init__(object_hook=datetime_pandas_object_hook, *args, **kwargs)


# https://chatgpt.com/c/69deac36-7300-8389-9441-1c7ce85b8722
def encode_state(obj):
    raw = json.dumps(obj, separators=(",", ":"), cls=DateTimePandasEncoder).encode()
    compressed = zlib.compress(raw)
    return base64.urlsafe_b64encode(compressed).decode().rstrip("=")

def decode_state(s):
    padding = "=" * (-len(s) % 4)
    compressed = base64.urlsafe_b64decode(s + padding)
    raw = zlib.decompress(compressed)
    return json.loads(raw, cls=DateTimePandasDecoder)

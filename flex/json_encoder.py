import json
import sys
from json.encoder import (
    INFINITY,
    _make_iterencode,
    c_make_encoder,
    encode_basestring,
    encode_basestring_ascii,
)
import importlib

import numpy as np
from astropy import units, coordinates, time
import datetime

INFINITY_VALUE = sys.float_info.max


class FlexJSONEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder for Flex files
    this overrides the float behaviour to default to null for nan values
    and the maximum floating point value for +/-infinity
    It also overrides the default method to handle numpy values and cast them 
    to their python base type
    """

    def iterencode(self, obj, _one_shot=False):
        """Encode the given object and yield each string
        representation as available.

        For example::

            for chunk in JSONEncoder().iterencode(bigobject):
                mysocket.write(chunk)

        """
        if self.check_circular:
            markers = {}
        else:
            markers = None
        if self.ensure_ascii:
            _encoder = encode_basestring_ascii
        else:
            _encoder = encode_basestring

        def floatstr(
            obj,
            allow_nan=self.allow_nan,
            _repr=float.__repr__,
            _inf=INFINITY,
            _neginf=-INFINITY,
        ):
            # Check for specials.  Note that this type of test is processor
            # and/or platform-specific, so do tests which don't depend on the
            # internals.

            if obj != obj:
                text = "null"
            elif obj == _inf:
                text = _repr(INFINITY_VALUE)
            elif obj == _neginf:
                text = _repr(-INFINITY_VALUE)
            else:
                return _repr(obj)

            return text

        if _one_shot and c_make_encoder is not None and self.indent is None:
            _iterencode = c_make_encoder(
                markers,
                self.default,
                _encoder,
                self.indent,
                self.key_separator,
                self.item_separator,
                self.sort_keys,
                self.skipkeys,
                self.allow_nan,
            )
        else:
            _iterencode = _make_iterencode(
                markers,
                self.default,
                _encoder,
                self.indent,
                floatstr,
                self.key_separator,
                self.item_separator,
                self.sort_keys,
                self.skipkeys,
                _one_shot,
            )
        return _iterencode(obj, 0)

    def default(self, obj):
        if hasattr(obj, "to_dict"):
            data = obj.to_dict()
            data["__module__"] = obj.__class__.__module__
            data["__class__"] = obj.__class__.__name__
            return data
        if isinstance(obj, coordinates.SkyCoord):
            return {
                "__module__": obj.__class__.__module__,
                "__class__": obj.__class__.__name__,
                "frame": obj.frame.name,
                "ra": self.default(obj.ra),
                "dec": self.default(obj.dec),
            }
        if isinstance(obj, coordinates.earth.EarthLocation):
            return {
                "__module__": obj.__class__.__module__,
                "__class__": obj.__class__.__name__,
                "x": self.default(obj.x),
                "y": self.default(obj.y),
                "z": self.default(obj.z),
            }
        if isinstance(obj, units.UnitBase):
            return {
                "__module__": obj.__class__.__module__,
                "__class__": "Unit",  # Unit can parse everything no problem
                "value": str(obj),
            }
        if isinstance(obj, units.quantity.Quantity):
            return {
                "__module__": obj.__class__.__module__,
                "__class__": obj.__class__.__name__,
                "value": obj.value,
                "unit": self.default(obj.unit),
            }
        if isinstance(obj, time.Time):
            return {
                "__module__": obj.__class__.__module__,
                "__class__": obj.__class__.__name__,
                "format": obj.format,
                "value": self.default(obj.value),
            }
        if isinstance(obj, datetime.datetime):
            return {
                "__module__": obj.__class__.__module__,
                "__class__": obj.__class__.__name__,
                "tzinfo": obj.tzinfo,
                "year": obj.year,
                "month": obj.month,
                "day": obj.day,
                "hour": obj.hour,
                "minute": obj.minute,
                "second": obj.second,
                "microsecond": obj.microsecond,
            }

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.str):
            return str(obj)

        return super().default(obj)


class FlexJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        kwargs["object_hook"] = self._object_hook
        super().__init__(*args, **kwargs)

    def _object_hook(self, obj):
        # If we specified a class and module use this:
        # but its not the lowest level of the header!
        if (
            isinstance(obj, dict)
            and "__class__" in obj
            and "__module__" in obj
            and not obj.get("__header__", False)
        ):
            module = importlib.import_module(obj["__module__"])
            cls = getattr(module, obj["__class__"])

            for k, v in obj.items():
                if k not in ["__class__", "__module__"]:
                    obj[k] = self._object_hook(v)

            args = (obj["value"],) if "value" in obj else ()
            exceptions = ["__class__", "__module__", "value"]
            kwargs = {k: v for k, v in obj.items() if k not in exceptions}

            if hasattr(cls, "from_json"):
                return cls.from_json(*args, **kwargs)

            return cls(*args, **kwargs)
        # Otherwise just return it
        return obj

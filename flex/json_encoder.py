import json
import sys
from json.encoder import (INFINITY, _make_iterencode, c_make_encoder,
                          encode_basestring, encode_basestring_ascii)

import numpy as np

INFINITY_VALUE = sys.float_info.max

class FlexJSONEncoder(json.JSONEncoder):
    """
    Custom JSON Encoder for Flex files
    this overrides the float behaviour to default to null for nan values
    and the maximum floating point value for +/-infinity
    It also overrides the default method to handle numpy values and cast them 
    to their python base type
    """
    def iterencode(self, o, _one_shot=False):
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

        def floatstr(o, allow_nan=self.allow_nan,
                _repr=float.__repr__, _inf=INFINITY, _neginf=-INFINITY):
            # Check for specials.  Note that this type of test is processor
            # and/or platform-specific, so do tests which don't depend on the
            # internals.

            if o != o:
                text = 'null'
            elif o == _inf:
                text = _repr(INFINITY_VALUE)
            elif o == _neginf:
                text = _repr(-INFINITY_VALUE)
            else:
                return _repr(o)

            return text


        if (_one_shot and c_make_encoder is not None
                and self.indent is None):
            _iterencode = c_make_encoder(
                markers, self.default, _encoder, self.indent,
                self.key_separator, self.item_separator, self.sort_keys,
                self.skipkeys, self.allow_nan)
        else:
            _iterencode = _make_iterencode(
                markers, self.default, _encoder, self.indent, floatstr,
                self.key_separator, self.item_separator, self.sort_keys,
                self.skipkeys, _one_shot)
        return _iterencode(o, 0)

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.str):
            return str(o)

        return super().default(o)


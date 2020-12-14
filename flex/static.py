import numpy as np
import pandas as pd

from .flex import FlexFile
from .extensions.bindata import BinaryDataExtension
from .extensions.tabledata import TableExtension


def write(filename, **data):
    header = {}
    extensions = {}
    for key, value in data.items():
        if hasattr(value, "__flex_save__"):
            extensions[key] = value
        elif type(value) is np.ndarray:
            extensions[key] = BinaryDataExtension(data=value)
        elif isinstance(value, pd.DataFrame):
            extensions[key] = TableExtension(data=value)
        else:
            header[key] = value

    ff = FlexFile(header=header, extensions=extensions)
    ff.write(filename)


def read(filename):
    return FlexFile.read(filename)

import pytest
import numpy as np
import pandas as pd

from fits2.fits2 import Fits2File
from fits2.extensions.tabledata import (
    TableExtension,
    AsciiTableExtension,
    JSONTableExtension,
)


@pytest.fixture
def table():
    df = pd.DataFrame(np.ones((10, 2)), columns=["A", "B"])
    return df


def test_read_write(tmp_fname, table):
    file = Fits2File()
    ext = TableExtension(data=table)
    file.extensions["tab"] = ext

    file.write(tmp_fname)

    del file
    f2 = Fits2File.read(tmp_fname)

    assert f2["tab"].data.size == 10 * 2
    assert np.all(f2["tab"].data == 1)
    assert "A" in f2["tab"].data.columns
    assert "B" in f2["tab"].data.columns


def test_read_write_ascii(tmp_fname, table):
    file = Fits2File()
    ext = AsciiTableExtension(data=table)
    file.extensions["tab"] = ext

    file.write(tmp_fname)

    del file
    f2 = Fits2File.read(tmp_fname)

    assert f2["tab"].data.size == 10 * 2
    assert np.all(f2["tab"].data == 1)
    assert "A" in f2["tab"].data.columns
    assert "B" in f2["tab"].data.columns


def test_read_write_json(tmp_fname, table):
    file = Fits2File()
    ext = JSONTableExtension(data=table)
    file.extensions["tab"] = ext

    file.write(tmp_fname)

    del file
    f2 = Fits2File.read(tmp_fname)

    assert f2["tab"].data.size == 10 * 2
    assert np.all(f2["tab"].data == 1)
    assert "A" in f2["tab"].data.columns
    assert "B" in f2["tab"].data.columns

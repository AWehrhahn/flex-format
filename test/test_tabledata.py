import pytest
import numpy as np
import pandas as pd

from flex.flex import FlexFile
from flex.extensions.tabledata import (
    TableExtension,
    AsciiTableExtension,
    JSONTableExtension,
)


@pytest.fixture
def table():
    df = pd.DataFrame(np.ones((10, 2)), columns=["A", "B"])
    return df


@pytest.fixture(
    params=[TableExtension, AsciiTableExtension, JSONTableExtension],
    ids=["table", "ascii", "json"],
)
def tableclass(request):
    return request.param


def test_read_write(tableclass, tmp_fname, table):
    file = FlexFile()
    ext = tableclass(data=table)
    file.extensions["tab"] = ext

    file.write(tmp_fname)

    del file
    f2 = FlexFile.read(tmp_fname)

    assert f2["tab"].data.size == 10 * 2
    assert np.all(f2["tab"].data == 1)
    assert len(f2["tab"].data.columns) == 2
    assert "A" in f2["tab"].data.columns
    assert "B" in f2["tab"].data.columns


def test_json(tableclass, tmp_fname, table):
    file = FlexFile()
    ext = tableclass(data=table)
    file.extensions["tab"] = ext

    file.to_json(tmp_fname)

    del file
    f2 = FlexFile.from_json(tmp_fname)

    assert f2["tab"].data.size == 10 * 2
    assert np.all(f2["tab"].data == 1)
    assert len(f2["tab"].data.columns) == 2
    assert "A" in f2["tab"].data.columns
    assert "B" in f2["tab"].data.columns

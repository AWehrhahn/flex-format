import numpy as np

from flex.flex import FlexFile


def test_read_write(tmp_fname):
    file = FlexFile()
    file.header["Hello"] = "World"
    file.write(tmp_fname)

    del file
    f2 = FlexFile.read(tmp_fname)

    assert isinstance(f2.header, dict)
    assert f2.header["Hello"] == "World"


def test_header_numpy_arrays(tmp_fname):
    file = FlexFile()
    file.header["Hello"] = np.zeros(3)
    file.write(tmp_fname)

    del file
    f2 = FlexFile.read(tmp_fname)

    assert isinstance(f2.header, dict)
    assert np.all(f2.header["Hello"] == 0)

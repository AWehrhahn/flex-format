import numpy as np

from fits2.fits2 import Fits2File


def test_read_write(tmp_fname):
    file = Fits2File()
    file.header["Hello"] = "World"
    file.write(tmp_fname)

    del file
    f2 = Fits2File.read(tmp_fname)

    assert isinstance(f2.header, dict)
    assert f2.header["Hello"] == "World"


def test_header_numpy_arrays(tmp_fname):
    file = Fits2File()
    file.header["Hello"] = np.zeros(3)
    file.write(tmp_fname)

    del file
    f2 = Fits2File.read(tmp_fname)

    assert isinstance(f2.header, dict)
    assert np.all(f2.header["Hello"] == 0)

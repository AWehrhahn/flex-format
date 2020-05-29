import numpy as np

from fits2.fits2 import Fits2File
from fits2.extensions.bindata import BinaryDataExtension


def test_read_write(tmp_fname):
    file = Fits2File()
    ext = BinaryDataExtension(data=np.zeros(10))
    file.extensions["img"] = ext

    file.write(tmp_fname)

    del file
    f2 = Fits2File.read(tmp_fname)

    assert f2["img"].data.size == 10
    assert np.all(f2["img"].data == 0)

import numpy as np

from flex.flex import FlexFile
from flex.extensions.bindata import BinaryDataExtension


def test_read_write(tmp_fname):
    file = FlexFile()
    ext = BinaryDataExtension(data=np.zeros(10))
    file.extensions["img"] = ext

    file.write(tmp_fname)

    del file
    f2 = FlexFile.read(tmp_fname)

    assert f2["img"].data.size == 10
    assert np.all(f2["img"].data == 0)

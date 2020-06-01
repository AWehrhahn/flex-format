import numpy as np

from flex.flex import FlexFile
from flex.extensions.bindata import BinaryDataExtension, MultipleDataExtension


def test_read_write(tmp_fname):
    file = FlexFile()
    ext = BinaryDataExtension(data=np.zeros(10))
    file.extensions["img"] = ext

    file.write(tmp_fname)

    del file
    f2 = FlexFile.read(tmp_fname)

    assert f2["img"].data.size == 10
    assert np.all(f2["img"].data == 0)


def test_empty_data(tmp_fname):
    file = FlexFile()
    ext = BinaryDataExtension(data=[])
    file.extensions["img"] = ext

    file.write(tmp_fname)

    del file
    f2 = FlexFile.read(tmp_fname)

    assert f2["img"].data.size == 0


def test_empty_data_json(tmp_fname):
    file = FlexFile()
    ext = BinaryDataExtension(data=[])
    file.extensions["img"] = ext

    file.to_json(tmp_fname)

    del file
    f2 = FlexFile.from_json(tmp_fname)

    assert f2["img"].data.size == 0


def test_json(tmp_fname):
    file = FlexFile()
    ext = BinaryDataExtension(data=np.zeros(10))
    file.extensions["img"] = ext

    file.to_json(tmp_fname)

    del file
    f2 = FlexFile.from_json(tmp_fname)

    assert f2["img"].data.size == 10
    assert np.all(f2["img"].data == 0)


def test_read_write_multi(tmp_fname):
    file = FlexFile()
    ext = MultipleDataExtension(data={"bla": np.zeros(10)})
    file.extensions["img"] = ext

    file.write(tmp_fname)

    del file
    f2 = FlexFile.read(tmp_fname)

    assert f2["img"].data["bla"].size == 10
    assert np.all(f2["img"].data["bla"] == 0)


def test_json_multi(tmp_fname):
    file = FlexFile()
    ext = MultipleDataExtension(data={"bla": np.zeros(10)})
    file.extensions["img"] = ext

    file.to_json(tmp_fname)

    del file
    f2 = FlexFile.from_json(tmp_fname)

    assert f2["img"].data["bla"].size == 10
    assert np.all(f2["img"].data["bla"] == 0)

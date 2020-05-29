from io import BytesIO
from os.path import join
from tarfile import TarInfo

from numpy.lib.format import open_memmap, read_magic, _read_array_header, _check_version

import numpy as np
import mmap

from ..fits2 import Fits2Extension


class BinaryDataExtension(Fits2Extension):
    def __init__(self, header={}, data=[]):
        super().__init__(header=header)
        self.data = np.asarray(data)

    @classmethod
    def _prepare_npy(cls, fname: str, data: np.ndarray):
        bio = BytesIO()
        np.save(bio, data)
        info = cls._get_tarinfo_from_bytesio(fname, bio)
        return info, bio

    def _prepare(self, name: str):
        cls = self.__class__
        header_fname = join(name, "header.json")
        data_fname = join(name, "data.npy")
        header_info, header_bio = cls._prepare_json(header_fname, self.header)
        data_info, data_bio = cls._prepare_npy(data_fname, self.data)

        return [(header_info, header_bio), (data_info, data_bio)]

    @staticmethod
    def _parse_npy(bio):
        mmapfile = bio.raw.fileobj
        if isinstance(mmapfile, mmap.mmap):
            version = read_magic(bio)
            _check_version(version)

            shape, fortran_order, dtype = _read_array_header(bio, version)
            if dtype.hasobject:
                msg = "Array can't be memory-mapped: Python objects in dtype."
                raise ValueError(msg)
            offset = bio.tell()
            order = "F" if fortran_order else "C"
            offset += 4 * 64  # WHY?
            data = np.ndarray.__new__(
                np.memmap,
                shape,
                dtype=dtype,
                buffer=mmapfile,
                offset=offset,
                order=order,
            )
            data._mmap = mmapfile
            data.offset = offset
            data.mode = "r+"
        else:
            b = BytesIO(bio.read())
            data = np.load(b)

        return data

    @classmethod
    def _read(cls, header: dict, members: dict):
        bio = members["data.npy"]
        data = cls._parse_npy(bio)
        ext = cls(header=header, data=data)
        return ext


class MultipleDataExtension(BinaryDataExtension):
    def __init__(self, header={}, data={}):
        super().__init__(header=header)
        self.data = dict(data)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def _prepare(self, name: str):
        cls = self.__class__

        header_fname = join(name, "header.json")
        header_info, header_bio = cls._prepare_json(header_fname, self.header)
        result = [(header_info, header_bio)]

        for key, value in self.data.items():
            data_fname = join(name, f"{key}.npy")
            data_info, data_bio = cls._prepare_npy(data_fname, value)
            result += [(data_info, data_bio)]

        return result

    @classmethod
    def _read(cls, header: dict, members: dict):
        data = {key: cls._parse_npy(bio) for key, bio in members.items()}
        ext = cls(header=header, data=data)
        return ext

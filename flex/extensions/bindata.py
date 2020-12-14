from io import BytesIO
from tarfile import TarInfo
import base64

from numpy.lib.format import open_memmap, read_magic, _read_array_header, _check_version

import numpy as np
import mmap

from ..flex import FlexExtension


class BinaryDataExtension(FlexExtension):
    def __init__(self, header={}, data=[], cls=None):
        super().__init__(header=header, cls=cls)
        self.data = np.asarray(data)

    @classmethod
    def _prepare_npy(cls, fname: str, data: np.ndarray):
        bio = BytesIO()
        np.save(bio, data)
        info = cls._get_tarinfo_from_bytesio(fname, bio)
        return info, bio

    def _prepare(self, name: str):
        cls = self.__class__
        header_fname = f"{name}/header.json"
        data_fname = f"{name}/data.npy"
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
            order = "F" if fortran_order else "C"
            offset = bio.tell()
            # Add the offset from the Wrapper file
            offset += bio.raw.offset
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
    def _parse(cls, header: dict, members: dict):
        bio = members["data.npy"]
        data = cls._parse_npy(bio)
        ext = cls(header=header, data=data)
        return ext

    @staticmethod
    def _np_to_dict(data):
        encoded = base64.b64encode(data.tobytes())
        encoded = encoded.decode("utf-8")
        obj = {
            "dtype": data.dtype.str,
            "shape": data.shape,
            "order": "C",
            "data": encoded,
        }
        return obj

    @staticmethod
    def _np_from_dict(data):
        decoded = data["data"].encode("utf-8")
        decoded = base64.b64decode(decoded)
        arr = np.frombuffer(decoded, dtype=data["dtype"])
        arr = arr.reshape(data["shape"])
        return arr

    def to_dict(self):
        cls = self.__class__
        obj = {
            "header": self.header,
            "data": cls._np_to_dict(self.data),
        }
        return obj

    @classmethod
    def from_dict(cls, header: dict, data: dict):
        arr = cls._np_from_dict(data["data"])
        obj = cls(header, arr)
        return obj


class MultipleDataExtension(BinaryDataExtension):
    def __init__(self, header={}, data={}, cls=None):
        super().__init__(header=header, cls=cls)
        self.data = dict(data)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def _prepare(self, name: str):
        cls = self.__class__

        header_fname = f"{name}/header.json"
        header_info, header_bio = cls._prepare_json(header_fname, self.header)
        result = [(header_info, header_bio)]

        for key, value in self.data.items():
            data_fname = f"{name}/{key}.npy"
            data_info, data_bio = cls._prepare_npy(data_fname, value)
            result += [(data_info, data_bio)]

        return result

    @classmethod
    def _parse(cls, header: dict, members: dict):
        data = {key[:-4]: cls._parse_npy(bio) for key, bio in members.items()}
        ext = cls(header=header, data=data)
        return ext

    def to_dict(self):
        cls = self.__class__
        obj = {"header": self.header}
        for name, data in self.data.items():
            obj[name] = cls._np_to_dict(data)
        return obj

    @classmethod
    def from_dict(cls, header: dict, data: dict):
        data = {name: cls._np_from_dict(d) for name, d in data.items()}
        obj = cls(header, data=data)
        return obj

from io import BytesIO
from os.path import join
from tarfile import TarInfo

import numpy as np

from ..fits2 import Fits2Extension


class BinaryDataExtension(Fits2Extension):
    def __init__(self, header={}, data=None):
        super().__init__(header=header)
        self.data = data

    @classmethod
    def _prepare_npy(cls, fname: str, data):
        bio = BytesIO()
        np.save(bio, data)
        info = cls._get_tarinfo_from_bytesio(fname, bio)
        return info, bio

    def _prepare(self, name: str):
        cls = self.__class__
        header_fname = join(name, "header.json")
        data_fname = join(name, "data.npy")
        header_info, header_bio = cls._prepare_json(
            header_fname, self.header
        )
        data_info, data_bio = cls._prepare_npy(data_fname, self.data)

        return [(header_info, header_bio), (data_info, data_bio)]

    @staticmethod
    def _parse_npy(bio):
        b = BytesIO(bio.read())
        data = np.load(b)
        return data

    @classmethod
    def _read(cls, header: dict, members: list):
        bio = members["data.npy"]
        data = cls._parse_npy(bio)
        ext = cls(header=header, data=data)
        return ext

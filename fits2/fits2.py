import importlib
import json
import tarfile
from io import BytesIO, TextIOWrapper
from os.path import dirname
from tarfile import TarFile, TarInfo

import numpy as np


class Fits2Base:
    @classmethod
    def _prepare_json(cls, fname: str, data: dict):
        tio = TextIOWrapper(BytesIO(), "utf-8")
        json.dump(data, tio)
        bio = tio.detach()
        info = cls._get_tarinfo_from_bytesio(fname, bio)
        return info, bio

    @staticmethod
    def _parse_json(bio):
        data = json.load(bio)
        return data

    @classmethod
    def _read_json(cls, file, name):
        bio = file.extractfile(name)
        data = cls._parse_json(bio)
        return data

    @staticmethod
    def _get_tarinfo_from_bytesio(fname, bio):
        info = TarInfo(fname)
        info.size = bio.tell()
        bio.seek(0)
        return info


class Fits2Extension(Fits2Base):
    def __init__(self, header={}, **kwargs):
        self.header = header
        self.header["extension_module"] = self.__class__.__module__
        self.header["extension_class"] = self.__class__.__name__

    def _prepare(self, name: str):
        raise NotImplementedError

    @staticmethod
    def _read(header, members):
        raise NotImplementedError


class Fits2File(Fits2Base):
    def __init__(self, header={}, extensions={}):
        self.header = header
        self.extensions = extensions

    def __getitem__(self, key):
        return self.extensions[key]

    def __setitem__(self, key, value):
        self.extensions[key] = value

    def write(self, fname: str):
        # Write the header
        cls = self.__class__
        info, bio = cls._prepare_json("header.json", self.header)
        extensions = []
        for name, ext in self.extensions.items():
            extensions += ext._prepare(name)
        with tarfile.open(fname, "w:gz") as file:
            file.addfile(info, bio)
            for ext in extensions:
                file.addfile(ext[0], ext[1])

    @staticmethod
    def read(fname: str):
        with tarfile.open(fname, "r") as file:
            header = Fits2File._read_json(file, "header.json")

            names = file.getnames()
            names = np.array([n for n in names if n != "header.json"])
            ext = [dirname(n) for n in names]
            ext, mapping = np.unique(ext, return_inverse=True)

            extensions = {}
            for i, name in enumerate(ext):
                # Determine the extension class and module
                ext_header = Fits2File._read_json(file, f"{name}\\header.json")
                ext_module = ext_header["extension_module"]
                ext_class = ext_header["extension_class"]

                ext_module = importlib.import_module(ext_module)
                ext_class = getattr(ext_module, ext_class)

                # Determine the files contributing to this extension
                members = names[mapping == i]
                ext_other = [n for n in members if not n.endswith("header.json")]

                ext_other = {
                    other[len(name) + 1 :]: file.extractfile(other)
                    for other in ext_other
                }
                exten = ext_class._read(ext_header, ext_other)

                extensions[name] = exten

        return Fits2File(header=header, extensions=extensions)

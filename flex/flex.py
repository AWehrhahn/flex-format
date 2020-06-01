import importlib
import json
import mmap
import tarfile
from io import BytesIO, TextIOWrapper
from os.path import dirname
from tarfile import TarFile, TarInfo

import numpy as np
from . import __version__


class FlexBase:
    @classmethod
    def _prepare_json(cls, fname: str, data: dict):
        tio = TextIOWrapper(BytesIO(), "utf-8")
        json.dump(data, tio, default=cls._to_base_type)
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

    @staticmethod
    def _to_base_type(value):
        if value is None:
            return value
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.str):
            return str(value)

        return value


class FlexExtension(FlexBase):
    def __init__(self, header={}, **kwargs):
        self.header = header
        self.header["extension_module"] = self.__class__.__module__
        self.header["extension_class"] = self.__class__.__name__

    def _prepare(self, name: str):
        raise NotImplementedError

    @classmethod
    def _parse(cls, header: dict, members: dict):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, header: dict, data):
        raise NotImplementedError


class FlexFile(FlexBase):
    def __init__(self, header={}, extensions={}):
        self.header = header
        self.extensions = extensions
        self.header["__version__"] = __version__

    def __getitem__(self, key):
        return self.extensions[key]

    def __setitem__(self, key, value):
        self.extensions[key] = value

    def write(self, fname: str, compression=False):
        # Write the header
        cls = self.__class__
        info, bio = cls._prepare_json("header.json", self.header)
        extensions = []
        for name, ext in self.extensions.items():
            extensions += ext._prepare(name)

        mode = "w:" if not compression else "w:gz"
        with tarfile.open(fname, mode) as file:
            file.addfile(info, bio)
            for ext in extensions:
                file.addfile(ext[0], ext[1])

    @classmethod
    def _read_ext_class(cls, ext_header):
        ext_module = ext_header["extension_module"]
        ext_class = ext_header["extension_class"]

        ext_module = importlib.import_module(ext_module)
        ext_class = getattr(ext_module, ext_class)
        return ext_class

    @classmethod
    def read(cls, fname: str):
        handle = open(fname, "rb")
        # If we allow mmap.ACCESS_WRITE, we invalidate the checksum
        # So I think the best solution is to use COPY
        # This also prevents the user accidentially messing up the files
        mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY)
        file = tarfile.open(mode="r", fileobj=mapped)
        header = cls._read_json(file, "header.json")

        names = file.getnames()
        names = np.array([n for n in names if n != "header.json"])
        ext = [dirname(n) for n in names]
        ext, mapping = np.unique(ext, return_inverse=True)

        extensions = {}
        for i, name in enumerate(ext):
            # Determine the files contributing to this extension
            members = names[mapping == i]
            ext_header = ""
            ext_other = []
            for n in members:
                if n.endswith("header.json"):
                    ext_header = n
                else:
                    ext_other += [n]

            # Determine the extension class and module
            ext_header = cls._read_json(file, ext_header)
            ext_class = cls._read_ext_class(ext_header)

            # TODO: lazy load the extensions?
            ext_other = {
                other[len(name) + 1 :]: file.extractfile(other) for other in ext_other
            }
            exten = ext_class._parse(ext_header, ext_other)

            extensions[name] = exten

        return cls(header=header, extensions=extensions)

    def to_dict(self):
        obj = {"header": self.header}
        for name, ext in self.extensions.items():
            obj[name] = ext.to_dict()
        return obj

    @classmethod
    def from_dict(cls, data: dict):
        extensions = {}
        for name, ext in data.items():
            if name == "header":
                header = ext
                continue

            ext_header = ext["header"]
            ext_class = cls._read_ext_class(ext_header)
            del ext["header"]
            exten = ext_class.from_dict(ext_header, ext)
            extensions[name] = exten

        obj = cls(header, extensions)
        return obj

    def to_json(self, fp=None):
        cls = self.__class__
        obj = self.to_dict()
        if fp is None:
            obj = json.dumps(obj, default=cls._to_base_type)
            return obj
        elif isinstance(fp, str):
            with open(fp, "w") as f:
                json.dump(obj, f, default=cls._to_base_type)
        else:
            json.dump(obj, fp, default=cls._to_base_type)

    @classmethod
    def from_json(cls, obj):
        try:
            obj = json.loads(obj)
        except json.decoder.JSONDecodeError as ex:
            # Its already a json string
            with open(obj, "r") as f:
                obj = json.load(f)
        obj = cls.from_dict(obj)
        return obj

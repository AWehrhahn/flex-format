import importlib
import json
import mmap
import tarfile
from io import BytesIO, TextIOWrapper
from os.path import dirname
from tarfile import TarFile, TarInfo

import numpy as np

from . import __version__
from .json_encoder import FlexJSONEncoder, FlexJSONDecoder


class FlexBase:
    @classmethod
    def _prepare_json(cls, fname: str, data: dict):
        tio = TextIOWrapper(BytesIO(), "utf-8")
        kwargs = dict(cls=FlexJSONEncoder, allow_nan=False, skipkeys=False)
        json.dump(data, tio, **kwargs)
        bio = tio.detach()
        info = cls._get_tarinfo_from_bytesio(fname, bio)
        return info, bio

    @staticmethod
    def _parse_json(bio):
        data = json.load(bio, cls=FlexJSONDecoder)
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


class FlexExtension(FlexBase):
    def __init__(self, header={}, cls=None, **kwargs):
        self.header = header
        if cls is not None:
            (self.header["__module__"], self.header["__class__"],) = cls.rsplit(".", 1)
        else:
            self.header["__module__"] = self.__class__.__module__
            self.header["__class__"] = self.__class__.__name__
        self.header["__header__"] = True

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


class JsonDataExtension(FlexExtension):
    def __init__(self, header={}, data={}, cls=None):
        super().__init__(header=header, cls=cls)
        self.data = data

    def _prepare(self, name: str):
        cls = self.__class__
        header_fname = f"{name}/header.json"
        data_fname = f"{name}/data.json"
        header_info, header_bio = cls._prepare_json(header_fname, self.header)
        data_info, data_bio = cls._prepare_json(data_fname, self.data)

        return [(header_info, header_bio), (data_info, data_bio)]

    @classmethod
    def _parse(cls, header: dict, members: dict):
        bio = members["data.json"]
        data = cls._parse_json(bio)
        ext = cls(header=header, data=data)
        return ext

    def to_dict(self):
        cls = self.__class__
        obj = {"header": self.header, "data": self.data}
        return obj

    @classmethod
    def from_dict(cls, header: dict, data: dict):
        arr = data["data"]
        obj = cls(header, arr)
        return obj


class FlexFile(FlexBase):
    def __init__(self, header={}, extensions={}, fileobj=None):
        if isinstance(header, str):
            other = self.read(header)
            self.header = other.header
            self.extensions = other.extensions
            self.fileobj = other.fileobj
        else:
            self.header = header
            self.extensions = extensions
            self.fileobj = fileobj
            self.header["__version__"] = __version__
            self.header["__header__"] = True

    def __getitem__(self, key):
        return self.extensions[key]

    def __setitem__(self, key, value):
        self.extensions[key] = value

    def write(self, fname: str, compression=False):
        # Write the header
        cls = self.__class__
        # Update header with current metadata
        self.header["__version__"] = __version__
        self.header["__header__"] = True

        info, bio = cls._prepare_json("header.json", self.header)
        extensions = []
        for name, ext in self.extensions.items():
            if isinstance(ext, FlexExtension):
                extensions += ext._prepare(name)
            elif hasattr(ext, "__flex_save__"):
                extensions += ext.__flex_save__()._prepare(name)
            elif hasattr(ext, "to_dict"):
                extensions += JsonDataExtension(
                    data=ext.to_dict(),
                    cls=f"{ext.__class__.__module__}.{ext.__class__.__name__}",
                )
            else:
                raise ValueError(f"Could not save extension {name}")

        mode = "w:" if not compression else "w:gz"
        with tarfile.open(fname, mode) as file:
            file.addfile(info, bio)
            for ext in extensions:
                file.addfile(ext[0], ext[1])

    @classmethod
    def _read_ext_class(cls, ext_header):
        try:
            ext_module = ext_header["__module__"]
            ext_class = ext_header["__class__"]
        except KeyError:
            ext_module = ext_header["extension_module"]
            ext_class = ext_header["extension_class"]

        ext_module = importlib.import_module(ext_module)
        ext_class = getattr(ext_module, ext_class)
        return ext_class

    @classmethod
    def read(cls, fname: str, mmap=False):
        handle = open(fname, "rb")
        # If we allow mmap.ACCESS_WRITE, we invalidate the checksum
        # So I think the best solution is to use COPY
        # This also prevents the user accidentially messing up the files
        if mmap:
            mapped = mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY)
            file = tarfile.open(mode="r", fileobj=mapped)
        else:
            file = tarfile.open(fname, mode="r")

        header = cls._read_json(file, "header.json")

        names = file.getnames()
        names = np.array([n for n in names if n != "header.json"])
        if len(names) != 0:
            # If the file was created using Windows style paths
            ext = np.char.replace(names, "\\", "/")
            ext = np.char.partition(ext, "/")
            ext = ext[:, 0]
            ext, mapping = np.unique(ext, return_inverse=True)
        else:
            ext, mapping = [], None

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
            if issubclass(ext_class, FlexExtension):
                exten = ext_class._parse(ext_header, ext_other)
            elif hasattr(ext_class, "__flex_load__"):
                exten = ext_class.__flex_load__(ext_header, ext_other)
            elif hasattr(ext_class, "from_dict"):
                temp_exten = JsonDataExtension._parse(ext_header, ext_other)
                exten = ext_class.from_dict(temp_exten.data)
            else:
                raise ValueError(f"Could not decode extension {name}")

            extensions[name] = exten

        return cls(header=header, extensions=extensions, fileobj=file)

    def close(self):
        if self.fileobj is not None:
            self.fileobj.fileobj.close()
            self.fileobj.close()

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
        kwargs = dict(cls=FlexJSONEncoder, allow_nan=False, skipkeys=False)
        if fp is None:
            obj = json.dumps(obj, **kwargs)
            return obj
        elif isinstance(fp, str):
            with open(fp, "w") as f:
                json.dump(obj, f, **kwargs)
        else:
            json.dump(obj, fp, **kwargs)

    @classmethod
    def from_json(cls, obj):
        try:
            obj = json.loads(obj, cls=FlexJSONDecoder)
        except json.decoder.JSONDecodeError as ex:
            # Its already a json string
            with open(obj, "r") as f:
                obj = json.load(f, cls=FlexJSONDecoder)
        obj = cls.from_dict(obj)
        return obj
